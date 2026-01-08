"""
Phase 5.2: Pools and Priority DAG

Demonstrates:
- Resource pools
- Task priority
- Concurrency control

"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.decorators import task
from airflow.operators.empty import EmptyOperator

default_args = {
    "owner": "airflow",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="phase5_02_pools_priority",
    description="Pools and priority demonstration",
    default_args=default_args,
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False,
    max_active_tasks=10,  # DAG-level concurrency limit
    tags=["phase5", "pools", "priority"],
    doc_md="""
    ## Pools and Priority DAG

    **Pools** limit concurrent task execution:
    - Prevent resource exhaustion
    - Control access to external systems
    - Share resources across DAGs

    **Priority** controls execution order:
    - Higher weight = higher priority
    - Affects scheduling order in queue

    **Setup pools via CLI:**
    ```bash
    # Create a pool with 3 slots
    airflow pools set ml_training_pool 3 "ML Training Resources"

    # Create database pool with 5 slots
    airflow pools set database_pool 5 "Database connections"
    ```
    """,
) as dag:

    start = EmptyOperator(task_id="start")

    # Tasks using a pool (create pool first!)
    # Each task takes 1 slot by default

    @task(
        pool="default_pool",  # Uses default pool if ml_training_pool doesn't exist
        pool_slots=1,  # Number of slots this task uses
        priority_weight=1,  # Lower priority
    )
    def low_priority_task(task_num: int):
        """Low priority task."""
        import time

        print(f"Low priority task {task_num} running...")
        time.sleep(2)
        return f"low_{task_num}"

    @task(
        pool="default_pool",
        pool_slots=1,
        priority_weight=10,  # Higher priority - runs first!
    )
    def high_priority_task(task_num: int):
        """High priority task - runs before low priority."""
        import time

        print(f"High priority task {task_num} running...")
        time.sleep(2)
        return f"high_{task_num}"

    @task(
        pool="default_pool",
        pool_slots=2,  # Uses 2 slots - limits concurrency
    )
    def resource_intensive_task():
        """Task that uses more pool slots."""
        import time

        print("Resource intensive task running (using 2 slots)...")
        time.sleep(5)
        return "intensive"

    # Create multiple instances
    low_tasks = [low_priority_task(i) for i in range(3)]
    high_tasks = [high_priority_task(i) for i in range(3)]
    intensive = resource_intensive_task()

    end = EmptyOperator(
        task_id="end",
        trigger_rule="all_done",
    )

    # All tasks after start, before end
    start >> low_tasks >> end
    start >> high_tasks >> end
    start >> intensive >> end
