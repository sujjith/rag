"""
Phase 4.1: XCom Basics DAG

Demonstrates:
- XCom push and pull
- Automatic return value XCom
- Multiple XCom values
- XCom with TaskFlow

"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.decorators import task

default_args = {
    "owner": "airflow",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


def push_value(**kwargs):
    """Push a value to XCom explicitly."""
    ti = kwargs["ti"]

    # Explicit push
    ti.xcom_push(key="explicit_value", value="Hello from explicit push!")

    # Multiple values
    ti.xcom_push(key="user_count", value=100)
    ti.xcom_push(key="status", value="active")

    # Return value is automatically pushed with key "return_value"
    return {"message": "Auto-pushed via return", "timestamp": str(datetime.now())}


def pull_value(**kwargs):
    """Pull XCom values from previous task."""
    ti = kwargs["ti"]

    # Pull explicit value
    explicit = ti.xcom_pull(task_ids="push_value", key="explicit_value")
    print(f"Explicit value: {explicit}")

    # Pull multiple values
    user_count = ti.xcom_pull(task_ids="push_value", key="user_count")
    status = ti.xcom_pull(task_ids="push_value", key="status")
    print(f"User count: {user_count}, Status: {status}")

    # Pull return value (default key)
    return_val = ti.xcom_pull(task_ids="push_value")  # key defaults to "return_value"
    print(f"Return value: {return_val}")

    return {"received_count": user_count, "received_status": status}


def aggregate_results(**kwargs):
    """Aggregate results from multiple tasks."""
    ti = kwargs["ti"]

    # Pull from multiple tasks at once
    results = ti.xcom_pull(task_ids=["push_value", "pull_value"])
    print(f"All results: {results}")

    # Access specific values
    push_result = ti.xcom_pull(task_ids="push_value")
    pull_result = ti.xcom_pull(task_ids="pull_value")

    return {
        "push_data": push_result,
        "pull_data": pull_result,
        "aggregated_at": str(datetime.now()),
    }


with DAG(
    dag_id="phase4_01_xcom_basics",
    description="XCom basics demonstration",
    default_args=default_args,
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False,
    tags=["phase4", "xcom"],
    doc_md="""
    ## XCom Basics DAG

    XCom (cross-communication) allows tasks to share small amounts of data.

    **Key Concepts:**
    - Return values are auto-pushed with key "return_value"
    - Use `xcom_push()` for explicit keys
    - Use `xcom_pull()` to retrieve values
    - XCom is stored in the Airflow database

    **Best Practices:**
    - Keep XCom values small (< 48KB recommended)
    - For large data, use external storage and pass references
    - Use TaskFlow API for cleaner XCom handling
    """,
) as dag:

    push = PythonOperator(
        task_id="push_value",
        python_callable=push_value,
    )

    pull = PythonOperator(
        task_id="pull_value",
        python_callable=pull_value,
    )

    aggregate = PythonOperator(
        task_id="aggregate_results",
        python_callable=aggregate_results,
    )

    push >> pull >> aggregate


# Bonus: TaskFlow XCom example
@task
def taskflow_producer():
    """TaskFlow automatically handles XCom."""
    return {"data": [1, 2, 3, 4, 5], "source": "taskflow"}


@task
def taskflow_consumer(data: dict):
    """Receives XCom automatically as parameter."""
    print(f"Received: {data}")
    total = sum(data["data"])
    return {"total": total, "source": data["source"]}
