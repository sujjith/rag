"""
Phase 7.1: Retry Strategies

Demonstrates enterprise retry patterns:
- Exponential backoff
- Conditional retries
- Custom retry logic
- Timeout enforcement
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.decorators import task
from airflow.operators.python import PythonOperator
from airflow.exceptions import AirflowException
import random


default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "email_on_failure": False,
    "retries": 3,
    "retry_delay": timedelta(seconds=10),
    # Exponential backoff - each retry waits longer
    "retry_exponential_backoff": True,
    "max_retry_delay": timedelta(minutes=5),
}


def exponential_backoff_task(**kwargs):
    """
    Task with exponential backoff retry.
    Retries: 10s -> 20s -> 40s -> 80s (capped at max_retry_delay)
    """
    attempt = kwargs["ti"].try_number
    print(f"Attempt {attempt} of task execution")

    # Simulate random failures (70% failure rate for demo)
    if random.random() < 0.7:
        raise AirflowException(f"Simulated failure on attempt {attempt}")

    print("Task succeeded!")
    return "success"


def conditional_retry_task(**kwargs):
    """
    Task that retries only on specific exceptions.
    In production, use on_retry_callback to implement custom logic.
    """
    attempt = kwargs["ti"].try_number

    # Simulate different error types
    error_type = random.choice(["transient", "permanent", "success"])

    if error_type == "transient":
        # Transient errors should retry
        raise AirflowException(f"Transient error on attempt {attempt} - will retry")
    elif error_type == "permanent":
        # Permanent errors - in production, you might want to skip retries
        # This requires custom exception handling
        raise ValueError(f"Permanent error - should not retry (but will due to Airflow default)")

    return "success"


def retry_callback(context):
    """Called before each retry attempt."""
    ti = context["ti"]
    exception = context.get("exception")

    print(f"Retry callback triggered for {ti.task_id}")
    print(f"Attempt: {ti.try_number}")
    print(f"Exception: {exception}")

    # Log to external system, send notification, etc.
    # In production: send to monitoring system


def timeout_task(**kwargs):
    """
    Task with execution timeout.
    Will be killed if exceeds execution_timeout.
    """
    import time

    # Simulate long-running task
    duration = random.randint(5, 15)
    print(f"Task will run for {duration} seconds")

    for i in range(duration):
        time.sleep(1)
        print(f"Working... {i+1}/{duration} seconds")

    return f"Completed in {duration} seconds"


with DAG(
    dag_id="phase7_01_retry_strategies",
    description="Enterprise retry patterns and strategies",
    default_args=default_args,
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False,
    tags=["phase7", "enterprise", "retry", "resilience"],
    doc_md="""
    ## Retry Strategies DAG

    Demonstrates enterprise-grade retry patterns:

    **Exponential Backoff:**
    - Increases wait time between retries
    - Prevents overwhelming failing services
    - Configured via `retry_exponential_backoff=True`

    **Retry Configuration:**
    - `retries`: Number of retry attempts
    - `retry_delay`: Initial delay between retries
    - `retry_exponential_backoff`: Enable exponential increase
    - `max_retry_delay`: Cap on retry delay

    **Timeout Enforcement:**
    - `execution_timeout`: Maximum task runtime
    - Task is killed if exceeded

    **Best Practices:**
    1. Always set timeouts for external calls
    2. Use exponential backoff for API calls
    3. Log retry attempts for debugging
    4. Consider circuit breakers for repeated failures
    """,
) as dag:

    # Task 1: Exponential backoff
    exp_backoff = PythonOperator(
        task_id="exponential_backoff_task",
        python_callable=exponential_backoff_task,
        retries=5,
        retry_delay=timedelta(seconds=5),
        retry_exponential_backoff=True,
        max_retry_delay=timedelta(minutes=2),
        on_retry_callback=retry_callback,
    )

    # Task 2: Conditional retry (with callback)
    conditional = PythonOperator(
        task_id="conditional_retry_task",
        python_callable=conditional_retry_task,
        retries=3,
        retry_delay=timedelta(seconds=5),
        on_retry_callback=retry_callback,
    )

    # Task 3: Task with timeout
    with_timeout = PythonOperator(
        task_id="timeout_task",
        python_callable=timeout_task,
        execution_timeout=timedelta(seconds=10),  # Kill if exceeds 10s
        retries=2,
        retry_delay=timedelta(seconds=5),
    )

    # Run independently to demonstrate each pattern
    [exp_backoff, conditional, with_timeout]
