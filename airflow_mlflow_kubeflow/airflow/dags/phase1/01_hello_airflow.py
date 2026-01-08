"""
Phase 1.1: Hello Airflow DAG

Your first Airflow DAG demonstrating basic concepts:
- DAG definition
- EmptyOperator
- Task dependencies

This DAG runs manually (schedule=None).
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.empty import EmptyOperator

# Default arguments for all tasks
default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

# DAG definition
with DAG(
    dag_id="phase1_01_hello_airflow",
    description="My first Airflow DAG",
    default_args=default_args,
    start_date=datetime(2024, 1, 1),
    schedule=None,  # Manual trigger only
    catchup=False,
    tags=["phase1", "example", "beginner"],
    doc_md="""
    ## Hello Airflow DAG

    This is your first Airflow DAG demonstrating:
    - Basic DAG structure
    - EmptyOperator usage
    - Simple task dependencies

    **How to run:**
    1. Enable the DAG in the UI
    2. Click "Trigger DAG" button
    3. Watch the tasks execute
    """,
) as dag:

    # Task definitions
    start = EmptyOperator(
        task_id="start",
        doc="Starting point of the DAG",
    )

    process = EmptyOperator(
        task_id="process",
        doc="Middle processing step",
    )

    end = EmptyOperator(
        task_id="end",
        doc="Ending point of the DAG",
    )

    # Task dependencies: start -> process -> end
    start >> process >> end
