"""
Phase 3.1: BashOperator DAG

Demonstrates:
- BashOperator usage
- Environment variables
- Templating in bash commands

"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator

default_args = {
    "owner": "airflow",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="phase3_01_bash_operator",
    description="BashOperator demonstrations",
    default_args=default_args,
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False,
    tags=["phase3", "operators", "bash"],
    doc_md="""
    ## BashOperator DAG

    Demonstrates various BashOperator features:
    - Simple commands
    - Environment variables
    - Jinja templating
    - Multi-line scripts
    """,
) as dag:

    # Simple bash command
    simple_bash = BashOperator(
        task_id="simple_bash",
        bash_command="echo 'Hello from BashOperator!'",
    )

    # With environment variables
    with_env = BashOperator(
        task_id="with_env_vars",
        bash_command="echo 'User: $USER, Var: $MY_VAR'",
        env={"MY_VAR": "custom_value"},
    )

    # Using Airflow templates
    templated = BashOperator(
        task_id="templated_command",
        bash_command="""
            echo "Execution date: {{ ds }}"
            echo "Previous date: {{ prev_ds }}"
            echo "Next date: {{ next_ds }}"
            echo "DAG ID: {{ dag.dag_id }}"
            echo "Task ID: {{ task.task_id }}"
        """,
    )

    # Multi-line script
    script = BashOperator(
        task_id="multi_line_script",
        bash_command="""
            echo "Starting script..."

            # Create some data
            for i in 1 2 3 4 5; do
                echo "Processing item $i"
                sleep 1
            done

            echo "Script complete!"
        """,
    )

    # Check Python version (useful for ML pipelines)
    check_python = BashOperator(
        task_id="check_python",
        bash_command="""
            echo "Python version:"
            python --version

            echo ""
            echo "Installed packages:"
            pip list | head -20
        """,
    )

    # Task dependencies
    simple_bash >> with_env >> templated >> script >> check_python
