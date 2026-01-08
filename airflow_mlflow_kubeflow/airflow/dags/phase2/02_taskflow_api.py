"""
Phase 2.2: TaskFlow API DAG

Demonstrates:
- Modern TaskFlow API with decorators
- Automatic XCom passing
- Clean, Pythonic DAG definition

"""
from datetime import datetime
from airflow.decorators import dag, task


@dag(
    dag_id="phase2_02_taskflow_api",
    description="Modern TaskFlow API demonstration",
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False,
    tags=["phase2", "taskflow", "modern"],
    doc_md="""
    ## TaskFlow API DAG

    The TaskFlow API is the modern, recommended way to write Airflow DAGs.

    **Benefits:**
    - Clean, Pythonic syntax
    - Automatic XCom handling
    - Type hints support
    - Less boilerplate

    **Key decorators:**
    - `@dag` - Define a DAG
    - `@task` - Define a task
    - `@task.branch` - Branching task
    - `@task_group` - Group tasks
    """,
)
def taskflow_demo():
    """
    A DAG demonstrating the TaskFlow API.
    """

    @task
    def extract():
        """Extract data from source."""
        print("Extracting data...")
        data = {
            "users": [
                {"id": 1, "name": "Alice", "score": 85},
                {"id": 2, "name": "Bob", "score": 92},
                {"id": 3, "name": "Charlie", "score": 78},
            ]
        }
        print(f"Extracted {len(data['users'])} users")
        return data

    @task
    def transform(data: dict) -> dict:
        """Transform the data."""
        print("Transforming data...")
        users = data["users"]

        # Calculate statistics
        scores = [u["score"] for u in users]
        result = {
            "total_users": len(users),
            "average_score": sum(scores) / len(scores),
            "max_score": max(scores),
            "min_score": min(scores),
        }

        print(f"Transformed: {result}")
        return result

    @task
    def load(result: dict):
        """Load the results."""
        print("Loading results...")
        print(f"Total Users: {result['total_users']}")
        print(f"Average Score: {result['average_score']:.2f}")
        print(f"Max Score: {result['max_score']}")
        print(f"Min Score: {result['min_score']}")
        print("Load complete!")

    # Define task flow - XCom is automatic!
    raw_data = extract()
    transformed_data = transform(raw_data)
    load(transformed_data)


# Instantiate the DAG
taskflow_demo()
