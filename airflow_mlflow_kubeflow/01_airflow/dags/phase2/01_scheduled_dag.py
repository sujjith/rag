"""
Phase 2.1: Scheduled DAG

Demonstrates:
- Cron-based scheduling
- Schedule presets
- Execution dates

"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator

default_args = {
    "owner": "airflow",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


def print_execution_info(**kwargs):
    """Print execution context information."""
    print("=" * 50)
    print("Execution Information")
    print("=" * 50)
    print(f"DAG ID: {kwargs['dag'].dag_id}")
    print(f"Task ID: {kwargs['task'].task_id}")
    print(f"Execution Date: {kwargs['ds']}")
    print(f"Logical Date: {kwargs['logical_date']}")
    print(f"Data Interval Start: {kwargs['data_interval_start']}")
    print(f"Data Interval End: {kwargs['data_interval_end']}")
    print(f"Run ID: {kwargs['run_id']}")
    print("=" * 50)


with DAG(
    dag_id="phase2_01_scheduled_dag",
    description="DAG with schedule demonstration",
    default_args=default_args,
    start_date=datetime(2024, 1, 1),
    # Schedule options:
    # "@once"    - Run once
    # "@hourly"  - Every hour at minute 0
    # "@daily"   - Every day at midnight
    # "@weekly"  - Every Sunday at midnight
    # "@monthly" - First day of month at midnight
    # "0 */6 * * *" - Every 6 hours (cron)
    # None       - Manual trigger only
    schedule="@daily",
    catchup=False,  # Don't backfill missed runs
    max_active_runs=1,  # Only one run at a time
    tags=["phase2", "scheduling"],
    doc_md="""
    ## Scheduled DAG

    This DAG runs on a schedule to demonstrate:
    - Schedule presets (@daily, @hourly, etc.)
    - Cron expressions
    - Execution date context

    **Schedule Options:**
    | Preset | Cron | Description |
    |--------|------|-------------|
    | @once | - | Run once |
    | @hourly | 0 * * * * | Every hour |
    | @daily | 0 0 * * * | Every day |
    | @weekly | 0 0 * * 0 | Every Sunday |
    | @monthly | 0 0 1 * * | First of month |
    """,
) as dag:

    print_info = PythonOperator(
        task_id="print_execution_info",
        python_callable=print_execution_info,
    )
