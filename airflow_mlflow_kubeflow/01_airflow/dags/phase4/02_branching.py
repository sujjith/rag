"""
Phase 4.2: Branching DAG

Demonstrates:
- BranchPythonOperator
- @task.branch decorator
- Multiple branch targets
- Branch skipping behavior

"""
from datetime import datetime, timedelta
import random
from airflow import DAG
from airflow.operators.python import BranchPythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.decorators import dag, task

default_args = {
    "owner": "airflow",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


def decide_branch(**kwargs):
    """Decide which branch to take based on conditions."""
    # Simulate different scenarios
    scenario = random.choice(["high", "medium", "low"])
    print(f"Scenario selected: {scenario}")

    # Return task_id(s) to execute
    if scenario == "high":
        return "process_high_priority"
    elif scenario == "medium":
        return "process_medium_priority"
    else:
        return "process_low_priority"


def decide_multiple_branches(**kwargs):
    """Return multiple branch targets."""
    value = random.randint(1, 10)
    kwargs["ti"].xcom_push(key="random_value", value=value)
    print(f"Value: {value}")

    branches = []
    if value > 3:
        branches.append("branch_a")
    if value > 5:
        branches.append("branch_b")
    if value > 7:
        branches.append("branch_c")

    # If no conditions met, take default branch
    if not branches:
        branches.append("branch_default")

    print(f"Selected branches: {branches}")
    return branches


with DAG(
    dag_id="phase4_02_branching",
    description="Branching workflow demonstration",
    default_args=default_args,
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False,
    tags=["phase4", "branching"],
    doc_md="""
    ## Branching DAG

    Demonstrates conditional execution paths in DAGs.

    **Key Concepts:**
    - `BranchPythonOperator` returns task_id(s) to execute
    - Non-selected branches are skipped
    - Can return multiple task_ids for parallel branches
    - Use `trigger_rule="none_failed_min_one_success"` for join tasks
    """,
) as dag:

    start = EmptyOperator(task_id="start")

    # Simple branching
    branch_decision = BranchPythonOperator(
        task_id="branch_decision",
        python_callable=decide_branch,
    )

    # Branch targets
    high_priority = EmptyOperator(task_id="process_high_priority")
    medium_priority = EmptyOperator(task_id="process_medium_priority")
    low_priority = EmptyOperator(task_id="process_low_priority")

    # Join point - must handle skipped tasks
    join = EmptyOperator(
        task_id="join",
        trigger_rule="none_failed_min_one_success",  # Important!
    )

    # Multi-branch decision
    multi_branch = BranchPythonOperator(
        task_id="multi_branch_decision",
        python_callable=decide_multiple_branches,
    )

    # Multiple possible targets
    branch_a = EmptyOperator(task_id="branch_a")
    branch_b = EmptyOperator(task_id="branch_b")
    branch_c = EmptyOperator(task_id="branch_c")
    branch_default = EmptyOperator(task_id="branch_default")

    # Final join
    final_join = EmptyOperator(
        task_id="final_join",
        trigger_rule="none_failed_min_one_success",
    )

    end = EmptyOperator(task_id="end")

    # Dependencies
    start >> branch_decision >> [high_priority, medium_priority, low_priority] >> join
    join >> multi_branch >> [branch_a, branch_b, branch_c, branch_default] >> final_join
    final_join >> end


# Modern TaskFlow branching example
@dag(
    dag_id="phase4_02b_taskflow_branching",
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False,
    tags=["phase4", "branching", "taskflow"],
)
def taskflow_branching_example():
    """TaskFlow API branching example."""

    @task
    def get_data():
        return {"value": random.randint(1, 100)}

    @task.branch
    def choose_path(data: dict):
        """Branch decorator for clean branching."""
        if data["value"] > 50:
            return "high_value_task"
        else:
            return "low_value_task"

    @task
    def high_value_task():
        print("Processing high value!")
        return "high"

    @task
    def low_value_task():
        print("Processing low value!")
        return "low"

    @task(trigger_rule="none_failed_min_one_success")
    def complete(result=None):
        print(f"Completed with: {result}")

    data = get_data()
    branch = choose_path(data)

    high_result = high_value_task()
    low_result = low_value_task()

    branch >> [high_result, low_result]
    [high_result, low_result] >> complete()


taskflow_branching_example()
