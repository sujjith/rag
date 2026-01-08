"""
Phase 1.2: Parallel Tasks DAG

Demonstrates:
- Parallel task execution
- Fan-out and fan-in patterns
- Task visualization in Graph view

"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.empty import EmptyOperator

default_args = {
    "owner": "airflow",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="phase1_02_parallel_tasks",
    description="Demonstrates parallel task execution",
    default_args=default_args,
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False,
    tags=["phase1", "example", "parallel"],
    doc_md="""
    ## Parallel Tasks DAG

    This DAG demonstrates:
    - **Fan-out**: One task leading to multiple parallel tasks
    - **Fan-in**: Multiple tasks converging to one task

    ```
         ┌─> task_a ─┐
    start ─┼─> task_b ─┼─> end
         └─> task_c ─┘
    ```
    """,
) as dag:

    # Start task
    start = EmptyOperator(task_id="start")

    # Parallel tasks (fan-out)
    task_a = EmptyOperator(task_id="task_a")
    task_b = EmptyOperator(task_id="task_b")
    task_c = EmptyOperator(task_id="task_c")

    # End task (fan-in)
    end = EmptyOperator(task_id="end")

    # Dependencies
    # start -> [task_a, task_b, task_c] -> end
    start >> [task_a, task_b, task_c] >> end
