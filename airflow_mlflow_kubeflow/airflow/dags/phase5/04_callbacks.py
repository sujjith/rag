"""
Phase 5.4: Callbacks and SLAs DAG

Demonstrates:
- Task callbacks (success, failure, retry)
- DAG callbacks
- SLA monitoring
- Custom alerting

"""
from datetime import datetime, timedelta
import random
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator

default_args = {
    "owner": "airflow",
    "retries": 2,
    "retry_delay": timedelta(seconds=10),
}


# Callback functions
def task_success_callback(context):
    """Called when task succeeds."""
    ti = context["task_instance"]
    print(f"SUCCESS: Task {ti.task_id} completed successfully!")
    print(f"  Execution date: {context['ds']}")
    print(f"  DAG: {context['dag'].dag_id}")

    # In production, you might:
    # - Send Slack notification
    # - Update monitoring dashboard
    # - Log to external system


def task_failure_callback(context):
    """Called when task fails (after all retries exhausted)."""
    ti = context["task_instance"]
    exception = context.get("exception")

    print(f"FAILURE: Task {ti.task_id} failed!")
    print(f"  Exception: {exception}")
    print(f"  Try number: {ti.try_number}")

    # In production:
    # - Send PagerDuty alert
    # - Create incident ticket
    # - Notify team via Slack/email


def task_retry_callback(context):
    """Called when task is retried."""
    ti = context["task_instance"]
    exception = context.get("exception")

    print(f"RETRY: Task {ti.task_id} is being retried")
    print(f"  Attempt: {ti.try_number}")
    print(f"  Exception: {exception}")


def sla_miss_callback(dag, task_list, blocking_task_list, slas, blocking_tis):
    """Called when SLA is missed."""
    print("SLA MISS DETECTED!")
    print(f"  DAG: {dag.dag_id}")
    print(f"  Tasks: {[t.task_id for t in task_list]}")
    print(f"  Blocking tasks: {blocking_task_list}")

    # In production:
    # - Alert on-call team
    # - Log SLA breach


def dag_success_callback(context):
    """Called when entire DAG succeeds."""
    dag_id = context["dag"].dag_id
    print(f"DAG SUCCESS: {dag_id} completed!")


def dag_failure_callback(context):
    """Called when DAG fails."""
    dag_id = context["dag"].dag_id
    print(f"DAG FAILURE: {dag_id} failed!")


# Task functions
def reliable_task(**kwargs):
    """Task that always succeeds."""
    print("Reliable task executing...")
    return "success"


def flaky_task(**kwargs):
    """Task that sometimes fails (to demonstrate retry callback)."""
    if random.random() < 0.5:
        raise Exception("Random failure - will retry")
    print("Flaky task succeeded!")
    return "success"


def slow_task(**kwargs):
    """Slow task that might miss SLA."""
    import time

    # This might cause SLA miss if > 30 seconds
    sleep_time = random.randint(5, 15)
    print(f"Slow task sleeping for {sleep_time} seconds...")
    time.sleep(sleep_time)
    return "completed"


with DAG(
    dag_id="phase5_04_callbacks",
    description="Callbacks and SLA demonstration",
    default_args=default_args,
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False,
    on_success_callback=dag_success_callback,
    on_failure_callback=dag_failure_callback,
    sla_miss_callback=sla_miss_callback,
    tags=["phase5", "callbacks", "sla"],
    doc_md="""
    ## Callbacks and SLA DAG

    **Task Callbacks:**
    - `on_success_callback` - Task succeeded
    - `on_failure_callback` - Task failed (after retries)
    - `on_retry_callback` - Task being retried

    **DAG Callbacks:**
    - `on_success_callback` - All tasks succeeded
    - `on_failure_callback` - Any task failed

    **SLA (Service Level Agreement):**
    - Define expected completion time
    - `sla_miss_callback` called on breach
    """,
) as dag:

    start = EmptyOperator(task_id="start")

    # Task with success callback
    reliable = PythonOperator(
        task_id="reliable_task",
        python_callable=reliable_task,
        on_success_callback=task_success_callback,
    )

    # Task with all callbacks
    flaky = PythonOperator(
        task_id="flaky_task",
        python_callable=flaky_task,
        on_success_callback=task_success_callback,
        on_failure_callback=task_failure_callback,
        on_retry_callback=task_retry_callback,
        retries=3,
    )

    # Task with SLA
    slow = PythonOperator(
        task_id="slow_task",
        python_callable=slow_task,
        sla=timedelta(seconds=30),  # SLA: must complete within 30 seconds
        on_success_callback=task_success_callback,
    )

    end = EmptyOperator(
        task_id="end",
        on_success_callback=task_success_callback,
        trigger_rule="all_done",
    )

    start >> reliable >> flaky >> slow >> end
