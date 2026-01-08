"""
Phase 4.3: Trigger Rules DAG

Demonstrates:
- All trigger rules
- Failure handling patterns
- Conditional execution

"""
from datetime import datetime, timedelta
import random
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.utils.trigger_rule import TriggerRule

default_args = {
    "owner": "airflow",
    "retries": 0,  # No retries to see trigger rules clearly
}


def success_task(**kwargs):
    """Task that always succeeds."""
    print("Success!")
    return "success"


def failing_task(**kwargs):
    """Task that always fails."""
    raise Exception("Intentional failure for demonstration")


def random_outcome(**kwargs):
    """Task with random success/failure."""
    if random.random() > 0.5:
        print("Random success!")
        return "success"
    else:
        raise Exception("Random failure!")


def cleanup_task(**kwargs):
    """Cleanup task that runs regardless of upstream status."""
    print("Running cleanup...")
    # This would contain cleanup logic
    return "cleaned"


with DAG(
    dag_id="phase4_03_trigger_rules",
    description="Trigger rules demonstration",
    default_args=default_args,
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False,
    tags=["phase4", "trigger_rules"],
    doc_md="""
    ## Trigger Rules DAG

    Trigger rules control when a task runs based on upstream task states.

    **Available Trigger Rules:**
    | Rule | Description |
    |------|-------------|
    | `all_success` | (Default) All parents succeeded |
    | `all_failed` | All parents failed |
    | `all_done` | All parents completed (any state) |
    | `one_success` | At least one parent succeeded |
    | `one_failed` | At least one parent failed |
    | `one_done` | At least one parent completed |
    | `none_failed` | No parent failed (success or skipped) |
    | `none_skipped` | No parent skipped |
    | `none_failed_min_one_success` | No failures and at least one success |
    | `always` | Run regardless of parent states |
    """,
) as dag:

    start = EmptyOperator(task_id="start")

    # Upstream tasks with different outcomes
    success_1 = PythonOperator(
        task_id="success_1",
        python_callable=success_task,
    )

    success_2 = PythonOperator(
        task_id="success_2",
        python_callable=success_task,
    )

    # This task will fail
    fail_task = PythonOperator(
        task_id="fail_task",
        python_callable=failing_task,
    )

    # Demonstrate different trigger rules

    # all_success (default) - runs only if ALL upstream succeed
    all_success = EmptyOperator(
        task_id="trigger_all_success",
        trigger_rule=TriggerRule.ALL_SUCCESS,
    )

    # all_failed - runs only if ALL upstream fail
    all_failed = EmptyOperator(
        task_id="trigger_all_failed",
        trigger_rule=TriggerRule.ALL_FAILED,
    )

    # all_done - runs when all upstream complete (regardless of status)
    all_done = EmptyOperator(
        task_id="trigger_all_done",
        trigger_rule=TriggerRule.ALL_DONE,
    )

    # one_success - runs if at least one upstream succeeds
    one_success = EmptyOperator(
        task_id="trigger_one_success",
        trigger_rule=TriggerRule.ONE_SUCCESS,
    )

    # one_failed - runs if at least one upstream fails
    one_failed = EmptyOperator(
        task_id="trigger_one_failed",
        trigger_rule=TriggerRule.ONE_FAILED,
    )

    # none_failed - runs if no upstream failed (success or skipped OK)
    none_failed = EmptyOperator(
        task_id="trigger_none_failed",
        trigger_rule=TriggerRule.NONE_FAILED,
    )

    # always - ALWAYS runs
    always_run = PythonOperator(
        task_id="always_cleanup",
        python_callable=cleanup_task,
        trigger_rule=TriggerRule.ALL_DONE,  # Use ALL_DONE for cleanup
    )

    end = EmptyOperator(
        task_id="end",
        trigger_rule=TriggerRule.ALL_DONE,
    )

    # Dependencies
    start >> [success_1, success_2, fail_task]

    [success_1, success_2] >> all_success
    [fail_task] >> all_failed
    [success_1, success_2, fail_task] >> all_done
    [success_1, success_2, fail_task] >> one_success
    [success_1, success_2, fail_task] >> one_failed
    [success_1, success_2] >> none_failed

    [all_success, all_failed, all_done, one_success, one_failed, none_failed] >> always_run >> end
