"""
Phase 5.1: Connections and Variables DAG

Demonstrates:
- Airflow Variables
- Airflow Connections
- Secure credential handling
- Configuration management

"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.models import Variable
from airflow.hooks.base import BaseHook

default_args = {
    "owner": "airflow",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


def use_variables(**kwargs):
    """Demonstrate Variable usage."""
    # Get single variable
    # Set these via UI: Admin > Variables
    # Or CLI: airflow variables set my_var "my_value"
    try:
        my_var = Variable.get("my_var", default_var="default_value")
        print(f"my_var: {my_var}")
    except Exception as e:
        print(f"Variable 'my_var' not set: {e}")
        my_var = "default_value"

    # Get JSON variable
    try:
        config = Variable.get("ml_config", deserialize_json=True, default_var={})
        print(f"ML Config: {config}")
    except Exception as e:
        print(f"Variable 'ml_config' not set, using defaults")
        config = {
            "learning_rate": 0.001,
            "epochs": 10,
            "batch_size": 32,
        }

    # Access nested values
    learning_rate = config.get("learning_rate", 0.001)
    print(f"Learning Rate: {learning_rate}")

    return {"variable_value": my_var, "config": config}


def use_connections(**kwargs):
    """Demonstrate Connection usage."""
    # Get connection details
    # Set via UI: Admin > Connections
    # Or CLI: airflow connections add ...

    try:
        # Get a connection (e.g., database connection)
        conn = BaseHook.get_connection("my_database")
        print(f"Host: {conn.host}")
        print(f"Port: {conn.port}")
        print(f"Schema: {conn.schema}")
        print(f"Login: {conn.login}")
        # conn.password is available but don't print it!

        # Get extra fields (JSON)
        extras = conn.extra_dejson
        print(f"Extra config: {extras}")

    except Exception as e:
        print(f"Connection 'my_database' not found: {e}")
        print("Create it via: Admin > Connections in Airflow UI")

    # Example: S3/MinIO connection
    try:
        s3_conn = BaseHook.get_connection("my_s3")
        print(f"S3 endpoint: {s3_conn.host}")
    except Exception as e:
        print(f"S3 connection not configured: {e}")

    return {"status": "connections_checked"}


def templated_variables(**kwargs):
    """Access variables via Jinja templates."""
    # Variables can also be accessed in templates:
    # {{ var.value.my_var }}
    # {{ var.json.ml_config }}

    # This function shows the programmatic approach
    ti = kwargs["ti"]

    # Pull results from previous tasks
    var_result = ti.xcom_pull(task_ids="use_variables")
    conn_result = ti.xcom_pull(task_ids="use_connections")

    print(f"Variable task result: {var_result}")
    print(f"Connection task result: {conn_result}")

    return "completed"


with DAG(
    dag_id="phase5_01_connections_variables",
    description="Connections and Variables demonstration",
    default_args=default_args,
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False,
    tags=["phase5", "connections", "variables"],
    doc_md="""
    ## Connections and Variables DAG

    **Variables** store configuration values:
    - Simple strings or JSON
    - Access via `Variable.get()` or templates
    - Store: model params, paths, feature flags

    **Connections** store external system credentials:
    - Database, S3, APIs, etc.
    - Encrypted password storage
    - Access via hooks or `BaseHook.get_connection()`

    **Setup:**
    ```bash
    # Set variable
    airflow variables set my_var "hello"
    airflow variables set ml_config '{"learning_rate": 0.001}'

    # Set connection
    airflow connections add my_database \\
        --conn-type postgres \\
        --conn-host localhost \\
        --conn-port 5432 \\
        --conn-login user \\
        --conn-password pass \\
        --conn-schema mydb
    ```
    """,
) as dag:

    variables_task = PythonOperator(
        task_id="use_variables",
        python_callable=use_variables,
    )

    connections_task = PythonOperator(
        task_id="use_connections",
        python_callable=use_connections,
    )

    summary = PythonOperator(
        task_id="summary",
        python_callable=templated_variables,
    )

    [variables_task, connections_task] >> summary
