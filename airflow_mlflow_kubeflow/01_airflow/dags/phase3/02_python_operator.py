"""
Phase 3.2: PythonOperator DAG

Demonstrates:
- PythonOperator usage
- Passing arguments
- Accessing context
- Return values (XCom)

"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator


def simple_function():
    """A simple function with no arguments."""
    print("Hello from Python!")
    return "simple_result"


def function_with_args(name, greeting="Hello"):
    """Function with positional and keyword arguments."""
    message = f"{greeting}, {name}!"
    print(message)
    return message


def function_with_context(**kwargs):
    """Function that uses Airflow context."""
    # Access task instance
    ti = kwargs["ti"]

    # Access execution context
    print(f"DAG ID: {kwargs['dag'].dag_id}")
    print(f"Task ID: {kwargs['task'].task_id}")
    print(f"Execution Date: {kwargs['ds']}")
    print(f"Run ID: {kwargs['run_id']}")

    # Pull XCom from previous task
    previous_result = ti.xcom_pull(task_ids="simple_python")
    print(f"Previous result: {previous_result}")

    return {"status": "success", "context_checked": True}


def data_processing_function(data_size, processing_type="standard"):
    """Simulate data processing."""
    import time
    import random

    print(f"Processing {data_size} records with {processing_type} method...")

    # Simulate processing
    time.sleep(2)
    processed_count = data_size - random.randint(0, 10)
    error_count = data_size - processed_count

    result = {
        "input_count": data_size,
        "processed_count": processed_count,
        "error_count": error_count,
        "success_rate": processed_count / data_size * 100,
    }

    print(f"Processing complete: {result}")
    return result


default_args = {
    "owner": "airflow",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="phase3_02_python_operator",
    description="PythonOperator demonstrations",
    default_args=default_args,
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False,
    tags=["phase3", "operators", "python"],
    doc_md="""
    ## PythonOperator DAG

    Demonstrates:
    - Simple Python functions
    - Passing arguments (op_args, op_kwargs)
    - Accessing Airflow context
    - XCom (cross-communication)
    """,
) as dag:

    # Simple function
    simple_python = PythonOperator(
        task_id="simple_python",
        python_callable=simple_function,
    )

    # With positional arguments
    with_args = PythonOperator(
        task_id="with_positional_args",
        python_callable=function_with_args,
        op_args=["Airflow"],  # Positional args
    )

    # With keyword arguments
    with_kwargs = PythonOperator(
        task_id="with_keyword_args",
        python_callable=function_with_args,
        op_kwargs={"name": "World", "greeting": "Hi"},
    )

    # With context
    with_context = PythonOperator(
        task_id="with_context",
        python_callable=function_with_context,
    )

    # Data processing simulation
    process_data = PythonOperator(
        task_id="process_data",
        python_callable=data_processing_function,
        op_kwargs={"data_size": 1000, "processing_type": "batch"},
    )

    # Dependencies
    simple_python >> with_args >> with_kwargs >> with_context >> process_data
