"""
Phase 8.1: Email Notifications

Demonstrates:
- EmailOperator usage
- Task failure notifications
- DAG completion summaries
- Custom email templates
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.decorators import task
from airflow.operators.email import EmailOperator
from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.utils.trigger_rule import TriggerRule


default_args = {
    "owner": "airflow",
    "retries": 1,
    "retry_delay": timedelta(minutes=1),
    # Email configuration (requires SMTP connection)
    "email": ["data-team@example.com"],
    "email_on_failure": True,
    "email_on_retry": False,
}


def task_failure_callback(context):
    """
    Custom callback for task failures.
    Sends detailed failure notification.
    """
    ti = context["ti"]
    dag_id = context["dag"].dag_id
    task_id = ti.task_id
    execution_date = context["execution_date"]
    exception = context.get("exception")
    log_url = ti.log_url

    subject = f"[AIRFLOW ALERT] Task Failed: {dag_id}.{task_id}"

    html_content = f"""
    <h2>Airflow Task Failure Alert</h2>

    <table border="1" cellpadding="5">
        <tr><td><b>DAG</b></td><td>{dag_id}</td></tr>
        <tr><td><b>Task</b></td><td>{task_id}</td></tr>
        <tr><td><b>Execution Date</b></td><td>{execution_date}</td></tr>
        <tr><td><b>Try Number</b></td><td>{ti.try_number}</td></tr>
        <tr><td><b>State</b></td><td>{ti.state}</td></tr>
    </table>

    <h3>Exception</h3>
    <pre>{exception}</pre>

    <p><a href="{log_url}">View Logs</a></p>

    <hr>
    <p><small>This is an automated alert from Apache Airflow</small></p>
    """

    print(f"Would send email: {subject}")
    print(f"To: {default_args['email']}")
    print(f"Content preview: Task {task_id} failed with: {exception}")

    # In production with SMTP configured:
    # from airflow.utils.email import send_email
    # send_email(
    #     to=default_args['email'],
    #     subject=subject,
    #     html_content=html_content,
    # )


def dag_success_callback(context):
    """Send summary email on DAG success."""
    dag_id = context["dag"].dag_id
    execution_date = context["execution_date"]

    subject = f"[AIRFLOW] DAG Completed: {dag_id}"

    html_content = f"""
    <h2>DAG Execution Completed Successfully</h2>

    <table border="1" cellpadding="5">
        <tr><td><b>DAG</b></td><td>{dag_id}</td></tr>
        <tr><td><b>Execution Date</b></td><td>{execution_date}</td></tr>
        <tr><td><b>Status</b></td><td style="color: green;">SUCCESS</td></tr>
    </table>

    <p>All tasks completed successfully.</p>
    """

    print(f"Would send success email: {subject}")


def simulate_task(**kwargs):
    """Simulated task that may fail."""
    import random
    task_id = kwargs["ti"].task_id

    # 20% failure rate for demo
    if random.random() < 0.2:
        raise Exception(f"Simulated failure in {task_id}")

    print(f"Task {task_id} completed successfully")
    return f"{task_id}_result"


with DAG(
    dag_id="phase8_01_email_notifications",
    description="Email notification patterns",
    default_args=default_args,
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False,
    on_success_callback=dag_success_callback,
    tags=["phase8", "enterprise", "notifications", "email"],
    doc_md="""
    ## Email Notifications

    **Prerequisites:**
    1. Configure SMTP connection in Airflow:
       - Admin -> Connections -> Add
       - Connection ID: `smtp_default`
       - Connection Type: Email
       - Host: smtp.example.com
       - Port: 587
       - Login: your-email
       - Password: your-password

    **Notification Types:**
    - Task failure alerts with details
    - DAG completion summaries
    - Custom formatted emails

    **Configuration:**
    - `email_on_failure`: Auto-send on task failure
    - `email_on_retry`: Send on retry attempts
    - Custom callbacks for detailed messages
    """,
) as dag:

    start = EmptyOperator(task_id="start")

    # Tasks with failure callbacks
    task_1 = PythonOperator(
        task_id="process_data",
        python_callable=simulate_task,
        on_failure_callback=task_failure_callback,
    )

    task_2 = PythonOperator(
        task_id="validate_data",
        python_callable=simulate_task,
        on_failure_callback=task_failure_callback,
    )

    task_3 = PythonOperator(
        task_id="load_data",
        python_callable=simulate_task,
        on_failure_callback=task_failure_callback,
    )

    # Explicit email notification task
    send_completion_email = EmailOperator(
        task_id="send_completion_email",
        to=["data-team@example.com"],
        subject="[Airflow] Pipeline {{ dag.dag_id }} completed - {{ ds }}",
        html_content="""
        <h2>Pipeline Execution Report</h2>

        <table border="1" cellpadding="5">
            <tr><td><b>DAG</b></td><td>{{ dag.dag_id }}</td></tr>
            <tr><td><b>Run ID</b></td><td>{{ run_id }}</td></tr>
            <tr><td><b>Execution Date</b></td><td>{{ ds }}</td></tr>
            <tr><td><b>Logical Date</b></td><td>{{ logical_date }}</td></tr>
        </table>

        <p>Pipeline completed. Check Airflow UI for details.</p>
        """,
        trigger_rule=TriggerRule.ALL_SUCCESS,
    )

    # Error notification task (runs on any failure)
    send_error_email = EmailOperator(
        task_id="send_error_notification",
        to=["oncall@example.com"],
        subject="[ALERT] Pipeline {{ dag.dag_id }} FAILED - {{ ds }}",
        html_content="""
        <h2 style="color: red;">Pipeline Failure Alert</h2>

        <p>One or more tasks in the pipeline have failed.</p>

        <table border="1" cellpadding="5">
            <tr><td><b>DAG</b></td><td>{{ dag.dag_id }}</td></tr>
            <tr><td><b>Execution Date</b></td><td>{{ ds }}</td></tr>
        </table>

        <p><b>Action Required:</b> Please investigate immediately.</p>
        """,
        trigger_rule=TriggerRule.ONE_FAILED,
    )

    end = EmptyOperator(
        task_id="end",
        trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS,
    )

    # Dependencies
    start >> [task_1, task_2, task_3]
    [task_1, task_2, task_3] >> send_completion_email >> end
    [task_1, task_2, task_3] >> send_error_email
