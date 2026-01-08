# Airflow Learning Guide - Phased Implementation

## Overview

This guide breaks down Apache Airflow into 6 learning phases.

| Phase | Topic | Outcome |
|-------|-------|---------|
| 1 | Setup & Basics | Airflow running, understand UI |
| 2 | DAG Fundamentals | Write and schedule DAGs |
| 3 | Operators & Sensors | Use various operators |
| 4 | XCom & Dependencies | Pass data, control flow |
| 5 | Advanced Features | Pools, hooks, connections |
| 6 | Integration | MLflow, Kubeflow pipelines |

---

# Phase 1: Setup & Basic Concepts

## Objectives
- Install Airflow on Kubernetes
- Navigate the Web UI
- Understand core concepts
- Use the CLI

## 1.1 Prerequisites

```bash
# Ensure Minikube is running (from Kubeflow setup)
minikube status

# Or start fresh
minikube start --cpus=4 --memory=8192

# Install Helm
curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash
```

## 1.2 Install Airflow

```bash
# Add Airflow Helm repo
helm repo add apache-airflow https://airflow.apache.org
helm repo update

# Create namespace
kubectl create namespace airflow

# Install Airflow
helm install airflow apache-airflow/airflow \
  --namespace airflow \
  -f helm/values.yaml \
  --timeout 10m

# Wait for pods
kubectl get pods -n airflow -w
```

## 1.3 Access the UI

```bash
# Port forward
kubectl port-forward svc/airflow-webserver -n airflow 8080:8080 &

# Open browser
echo "Airflow UI: http://localhost:8080"
echo "Username: admin"
echo "Password: admin"
```

## 1.4 Understanding the UI

### Home Page
- List of all DAGs
- Toggle DAGs on/off
- Quick status overview

### DAG View
- Graph view: Visual task dependencies
- Tree view: Historical runs
- Calendar view: Schedule overview
- Code view: DAG source code

### Admin Menu
- Connections: External system credentials
- Variables: Key-value configuration
- Pools: Resource limits
- XComs: Task communication data

## 1.5 CLI Basics

```bash
# Enter webserver pod
kubectl exec -it -n airflow deployment/airflow-webserver -- bash

# List DAGs
airflow dags list

# Show DAG details
airflow dags show <dag_id>

# Trigger DAG
airflow dags trigger <dag_id>

# List tasks in DAG
airflow tasks list <dag_id>

# Test task
airflow tasks test <dag_id> <task_id> <execution_date>

# Check scheduler
airflow jobs check
```

## Phase 1 Checklist

- [ ] Airflow installed on Kubernetes
- [ ] Web UI accessible
- [ ] Understand main UI sections
- [ ] Can use basic CLI commands

---

# Phase 2: DAG Fundamentals

## Objectives
- Write your first DAG
- Understand scheduling
- Configure DAG parameters

## 2.1 DAG Structure

```python
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.empty import EmptyOperator

# DAG definition
with DAG(
    dag_id="my_first_dag",
    description="My first Airflow DAG",
    start_date=datetime(2024, 1, 1),
    schedule="@daily",  # or cron: "0 0 * * *"
    catchup=False,
    tags=["example", "phase2"],
    default_args={
        "owner": "airflow",
        "retries": 1,
        "retry_delay": timedelta(minutes=5),
    },
) as dag:

    # Tasks
    start = EmptyOperator(task_id="start")
    end = EmptyOperator(task_id="end")

    # Dependencies
    start >> end
```

## 2.2 Schedule Intervals

| Preset | Cron | Description |
|--------|------|-------------|
| `@once` | - | Run once |
| `@hourly` | `0 * * * *` | Every hour |
| `@daily` | `0 0 * * *` | Every day at midnight |
| `@weekly` | `0 0 * * 0` | Every Sunday |
| `@monthly` | `0 0 1 * *` | First day of month |
| `@yearly` | `0 0 1 1 *` | January 1st |
| `None` | - | Manual trigger only |

## 2.3 DAG Parameters

```python
with DAG(
    dag_id="parameterized_dag",
    start_date=datetime(2024, 1, 1),
    schedule="@daily",

    # Important parameters
    catchup=False,           # Don't backfill
    max_active_runs=1,       # One run at a time
    concurrency=4,           # Max parallel tasks
    dagrun_timeout=timedelta(hours=2),

    # Default task arguments
    default_args={
        "owner": "data-team",
        "depends_on_past": False,
        "email": ["alerts@example.com"],
        "email_on_failure": True,
        "email_on_retry": False,
        "retries": 2,
        "retry_delay": timedelta(minutes=5),
        "execution_timeout": timedelta(hours=1),
    },

    # Documentation
    doc_md="""
    ## My DAG Documentation
    This DAG does important things.
    """,

    tags=["production", "etl"],
) as dag:
    pass
```

## 2.4 Task Dependencies

```python
from airflow.operators.empty import EmptyOperator

# Create tasks
task_a = EmptyOperator(task_id="task_a")
task_b = EmptyOperator(task_id="task_b")
task_c = EmptyOperator(task_id="task_c")
task_d = EmptyOperator(task_id="task_d")

# Linear: A -> B -> C
task_a >> task_b >> task_c

# Parallel: A -> [B, C] -> D
task_a >> [task_b, task_c] >> task_d

# Using set_downstream/set_upstream
task_a.set_downstream(task_b)
task_c.set_upstream(task_b)

# Complex dependencies
from airflow.models.baseoperator import chain, cross_downstream

# Chain: A -> B -> C -> D
chain(task_a, task_b, task_c, task_d)

# Cross downstream: [A, B] -> [C, D]
cross_downstream([task_a, task_b], [task_c, task_d])
```

## Phase 2 Checklist

- [ ] Created first DAG
- [ ] Understand scheduling options
- [ ] Can set task dependencies
- [ ] DAG appears in UI

---

# Phase 3: Operators & Sensors

## Objectives
- Use common operators
- Create custom operators
- Implement sensors

## 3.1 BashOperator

```python
from airflow.operators.bash import BashOperator

# Simple command
bash_task = BashOperator(
    task_id="run_bash",
    bash_command="echo 'Hello from Bash!'",
)

# With environment variables
env_task = BashOperator(
    task_id="with_env",
    bash_command="echo $MY_VAR",
    env={"MY_VAR": "Hello World"},
)

# Run script
script_task = BashOperator(
    task_id="run_script",
    bash_command="/opt/scripts/my_script.sh ",
)

# With templating
template_task = BashOperator(
    task_id="templated",
    bash_command="echo 'Execution date: {{ ds }}'",
)
```

## 3.2 PythonOperator

```python
from airflow.operators.python import PythonOperator

def my_function(name, **kwargs):
    """Python function to execute."""
    print(f"Hello, {name}!")
    print(f"Execution date: {kwargs['ds']}")
    return "Success"

python_task = PythonOperator(
    task_id="run_python",
    python_callable=my_function,
    op_kwargs={"name": "Airflow"},
)

# With op_args
def add_numbers(a, b):
    return a + b

add_task = PythonOperator(
    task_id="add_numbers",
    python_callable=add_numbers,
    op_args=[10, 20],
)
```

## 3.3 TaskFlow API (Recommended)

```python
from airflow.decorators import dag, task
from datetime import datetime

@dag(
    dag_id="taskflow_example",
    start_date=datetime(2024, 1, 1),
    schedule="@daily",
    catchup=False,
)
def taskflow_dag():

    @task
    def extract():
        """Extract data."""
        return {"data": [1, 2, 3, 4, 5]}

    @task
    def transform(data: dict):
        """Transform data."""
        return {"result": sum(data["data"])}

    @task
    def load(result: dict):
        """Load data."""
        print(f"Result: {result['result']}")

    # Automatic XCom passing
    data = extract()
    result = transform(data)
    load(result)

# Instantiate DAG
taskflow_dag()
```

## 3.4 Sensors

```python
from airflow.sensors.filesystem import FileSensor
from airflow.sensors.python import PythonSensor
from airflow.sensors.time_delta import TimeDeltaSensor
from datetime import timedelta

# Wait for file
file_sensor = FileSensor(
    task_id="wait_for_file",
    filepath="/data/input/file.csv",
    poke_interval=60,  # Check every 60 seconds
    timeout=3600,      # Timeout after 1 hour
    mode="poke",       # or "reschedule"
)

# Wait for condition
def check_condition(**kwargs):
    # Return True when condition is met
    import random
    return random.random() > 0.7

condition_sensor = PythonSensor(
    task_id="wait_for_condition",
    python_callable=check_condition,
    poke_interval=30,
    timeout=600,
)

# Wait for time delta
wait_sensor = TimeDeltaSensor(
    task_id="wait_5_minutes",
    delta=timedelta(minutes=5),
)
```

## 3.5 Common Operators

```python
from airflow.operators.email import EmailOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.providers.http.operators.http import SimpleHttpOperator

# Email
email_task = EmailOperator(
    task_id="send_email",
    to="user@example.com",
    subject="Airflow Alert",
    html_content="<h1>Task completed!</h1>",
)

# Trigger another DAG
trigger_task = TriggerDagRunOperator(
    task_id="trigger_other_dag",
    trigger_dag_id="other_dag",
    wait_for_completion=True,
)

# HTTP request
http_task = SimpleHttpOperator(
    task_id="call_api",
    http_conn_id="my_api",
    endpoint="/api/data",
    method="GET",
)
```

## Phase 3 Checklist

- [ ] Used BashOperator
- [ ] Used PythonOperator
- [ ] Used TaskFlow API
- [ ] Implemented sensors

---

# Phase 4: XCom & Dependencies

## Objectives
- Pass data between tasks
- Implement branching
- Control task flow

## 4.1 XCom Basics

```python
from airflow.decorators import dag, task
from datetime import datetime

@dag(dag_id="xcom_example", start_date=datetime(2024, 1, 1), schedule=None)
def xcom_dag():

    @task
    def produce_data():
        """Push data to XCom."""
        return {"value": 42, "message": "Hello"}

    @task
    def consume_data(data: dict):
        """Pull data from XCom."""
        print(f"Received: {data}")
        return data["value"] * 2

    @task
    def final_task(result: int):
        print(f"Final result: {result}")

    data = produce_data()
    result = consume_data(data)
    final_task(result)

xcom_dag()
```

## 4.2 Manual XCom

```python
from airflow.operators.python import PythonOperator

def push_function(**kwargs):
    """Push value to XCom."""
    ti = kwargs["ti"]
    ti.xcom_push(key="my_key", value="my_value")

def pull_function(**kwargs):
    """Pull value from XCom."""
    ti = kwargs["ti"]
    value = ti.xcom_pull(task_ids="push_task", key="my_key")
    print(f"Pulled: {value}")

push_task = PythonOperator(
    task_id="push_task",
    python_callable=push_function,
)

pull_task = PythonOperator(
    task_id="pull_task",
    python_callable=pull_function,
)

push_task >> pull_task
```

## 4.3 Branching

```python
from airflow.decorators import dag, task
from airflow.operators.empty import EmptyOperator
from datetime import datetime
import random

@dag(dag_id="branching_example", start_date=datetime(2024, 1, 1), schedule=None)
def branching_dag():

    @task.branch
    def choose_branch():
        """Decide which branch to take."""
        value = random.random()
        if value > 0.5:
            return "high_path"
        else:
            return "low_path"

    @task
    def high_path():
        print("Taking high path!")

    @task
    def low_path():
        print("Taking low path!")

    @task(trigger_rule="none_failed_min_one_success")
    def join():
        print("Joined!")

    branch = choose_branch()
    high = high_path()
    low = low_path()
    end = join()

    branch >> [high, low] >> end

branching_dag()
```

## 4.4 Trigger Rules

```python
from airflow.operators.empty import EmptyOperator
from airflow.utils.trigger_rule import TriggerRule

# Available trigger rules:
# - all_success (default): All parents succeeded
# - all_failed: All parents failed
# - all_done: All parents completed
# - one_success: At least one parent succeeded
# - one_failed: At least one parent failed
# - none_failed: No parent failed
# - none_skipped: No parent skipped
# - none_failed_min_one_success: No failures and at least one success

task_always = EmptyOperator(
    task_id="always_run",
    trigger_rule=TriggerRule.ALL_DONE,
)

task_on_failure = EmptyOperator(
    task_id="on_failure",
    trigger_rule=TriggerRule.ONE_FAILED,
)
```

## 4.5 Dynamic Task Mapping

```python
from airflow.decorators import dag, task
from datetime import datetime

@dag(dag_id="dynamic_mapping", start_date=datetime(2024, 1, 1), schedule=None)
def dynamic_dag():

    @task
    def generate_list():
        return [1, 2, 3, 4, 5]

    @task
    def process_item(item: int):
        return item * 2

    @task
    def aggregate(results: list):
        print(f"Sum: {sum(results)}")

    items = generate_list()
    processed = process_item.expand(item=items)  # Maps over list
    aggregate(processed)

dynamic_dag()
```

## Phase 4 Checklist

- [ ] Passed data via XCom
- [ ] Implemented branching
- [ ] Used trigger rules
- [ ] Created dynamic tasks

---

# Phase 5: Advanced Features

## Objectives
- Use connections and variables
- Implement pools
- Create custom hooks

## 5.1 Connections

```python
from airflow.hooks.base import BaseHook

# Get connection from Airflow
conn = BaseHook.get_connection("my_database")
print(f"Host: {conn.host}")
print(f"Port: {conn.port}")
print(f"Login: {conn.login}")

# Use in operator
from airflow.providers.postgres.operators.postgres import PostgresOperator

sql_task = PostgresOperator(
    task_id="run_sql",
    postgres_conn_id="my_database",
    sql="SELECT * FROM users LIMIT 10",
)
```

## 5.2 Variables

```python
from airflow.models import Variable

# Get variable
my_var = Variable.get("my_variable")
my_json = Variable.get("my_json_var", deserialize_json=True)
with_default = Variable.get("missing_var", default_var="default")

# Set variable (usually via UI or CLI)
Variable.set("my_variable", "my_value")

# In templates
# {{ var.value.my_variable }}
# {{ var.json.my_json_var }}
```

## 5.3 Pools

```python
from airflow.operators.python import PythonOperator

# Pools limit concurrent task execution
# Create pools in Admin -> Pools

limited_task = PythonOperator(
    task_id="limited_task",
    python_callable=my_function,
    pool="limited_pool",      # Pool name
    pool_slots=1,             # Slots to use
    priority_weight=10,       # Higher = higher priority
)
```

## 5.4 Task Groups

```python
from airflow.decorators import dag, task, task_group
from datetime import datetime

@dag(dag_id="task_groups_example", start_date=datetime(2024, 1, 1), schedule=None)
def task_groups_dag():

    @task
    def start():
        return "Starting"

    @task_group(group_id="extract_group")
    def extract_tasks():
        @task
        def extract_users():
            return {"users": 100}

        @task
        def extract_orders():
            return {"orders": 500}

        return extract_users(), extract_orders()

    @task_group(group_id="transform_group")
    def transform_tasks(data):
        @task
        def transform(item):
            return item

        return transform.expand(item=data)

    @task
    def load(data):
        print(f"Loading: {data}")

    start_task = start()
    extracted = extract_tasks()
    transformed = transform_tasks(extracted)
    load(transformed)

task_groups_dag()
```

## 5.5 Callbacks

```python
def success_callback(context):
    """Called on task success."""
    print(f"Task {context['task_instance'].task_id} succeeded!")

def failure_callback(context):
    """Called on task failure."""
    print(f"Task {context['task_instance'].task_id} failed!")
    # Send alert, etc.

def sla_miss_callback(dag, task_list, blocking_task_list, slas, blocking_tis):
    """Called when SLA is missed."""
    print("SLA missed!")

with DAG(
    dag_id="callbacks_example",
    on_success_callback=success_callback,
    on_failure_callback=failure_callback,
    sla_miss_callback=sla_miss_callback,
    # ...
) as dag:

    task_with_callbacks = PythonOperator(
        task_id="my_task",
        python_callable=my_function,
        on_success_callback=success_callback,
        on_failure_callback=failure_callback,
    )
```

## Phase 5 Checklist

- [ ] Configured connections
- [ ] Used variables
- [ ] Implemented pools
- [ ] Created task groups
- [ ] Used callbacks

---

# Phase 6: Integration

## Objectives
- Trigger Kubeflow pipelines
- Log to MLflow
- Build end-to-end workflows

## 6.1 MLflow Integration

```python
from airflow.decorators import dag, task
from datetime import datetime

@dag(dag_id="mlflow_training", start_date=datetime(2024, 1, 1), schedule=None)
def mlflow_training_dag():

    @task
    def train_model():
        import mlflow
        import mlflow.sklearn
        from sklearn.datasets import load_iris
        from sklearn.ensemble import RandomForestClassifier

        mlflow.set_tracking_uri("http://mlflow-server:5000")
        mlflow.set_experiment("airflow-experiments")

        iris = load_iris()
        with mlflow.start_run():
            model = RandomForestClassifier(n_estimators=100)
            model.fit(iris.data, iris.target)

            mlflow.log_param("n_estimators", 100)
            mlflow.sklearn.log_model(model, "model")

            return mlflow.active_run().info.run_id

    @task
    def register_model(run_id: str):
        import mlflow
        mlflow.set_tracking_uri("http://mlflow-server:5000")
        mlflow.register_model(f"runs:/{run_id}/model", "airflow-model")

    run_id = train_model()
    register_model(run_id)

mlflow_training_dag()
```

## 6.2 Kubeflow Pipeline Trigger

```python
from airflow.decorators import dag, task
from airflow.operators.python import PythonOperator
from datetime import datetime

@dag(dag_id="trigger_kubeflow", start_date=datetime(2024, 1, 1), schedule=None)
def kubeflow_trigger_dag():

    @task
    def trigger_pipeline():
        import kfp
        client = kfp.Client(host="http://ml-pipeline-ui.kubeflow:80")

        run = client.create_run_from_pipeline_package(
            pipeline_file="training_pipeline.yaml",
            arguments={"n_estimators": 100},
            run_name="airflow-triggered"
        )
        return run.run_id

    @task
    def wait_for_completion(run_id: str):
        import kfp
        import time

        client = kfp.Client(host="http://ml-pipeline-ui.kubeflow:80")

        while True:
            run = client.get_run(run_id)
            if run.run.status in ["Succeeded", "Failed", "Error"]:
                return run.run.status
            time.sleep(30)

    run_id = trigger_pipeline()
    status = wait_for_completion(run_id)

kubeflow_trigger_dag()
```

## 6.3 Complete ML Pipeline DAG

```python
from airflow.decorators import dag, task, task_group
from airflow.operators.python import BranchPythonOperator
from datetime import datetime

@dag(
    dag_id="complete_ml_pipeline",
    start_date=datetime(2024, 1, 1),
    schedule="@daily",
    catchup=False,
)
def ml_pipeline():

    @task_group(group_id="data_preparation")
    def data_prep():
        @task
        def extract_data():
            return {"source": "database", "rows": 10000}

        @task
        def validate_data(data: dict):
            # Run Great Expectations or similar
            return {"valid": True, "rows": data["rows"]}

        @task
        def transform_data(data: dict):
            return {"features": 50, "samples": data["rows"]}

        extracted = extract_data()
        validated = validate_data(extracted)
        return transform_data(validated)

    @task_group(group_id="model_training")
    def model_training(data: dict):
        @task
        def train_model(data: dict):
            # Train and log to MLflow
            return {"model_id": "model-123", "accuracy": 0.95}

        @task
        def evaluate_model(model: dict):
            return model["accuracy"] > 0.9

        model = train_model(data)
        return evaluate_model(model)

    @task.branch
    def check_performance(passed: bool):
        if passed:
            return "deploy_model"
        return "notify_failure"

    @task
    def deploy_model():
        # Deploy to KServe/Seldon
        return "deployed"

    @task
    def notify_failure():
        # Send alert
        return "notified"

    @task(trigger_rule="none_failed_min_one_success")
    def cleanup():
        print("Pipeline completed")

    data = data_prep()
    passed = model_training(data)
    branch = check_performance(passed)

    deploy = deploy_model()
    notify = notify_failure()

    branch >> [deploy, notify] >> cleanup()

ml_pipeline()
```

## Phase 6 Checklist

- [ ] Integrated with MLflow
- [ ] Triggered Kubeflow pipelines
- [ ] Built end-to-end ML workflow

---

# Summary

## Complete Learning Path

| Phase | Topic | Key Takeaways |
|-------|-------|---------------|
| 1 | Setup | Installation, UI, CLI |
| 2 | DAG Fundamentals | Structure, scheduling |
| 3 | Operators | Bash, Python, Sensors |
| 4 | XCom | Data passing, branching |
| 5 | Advanced | Pools, connections, groups |
| 6 | Integration | MLflow, Kubeflow |

## Quick Reference

```bash
# Access UI
kubectl port-forward svc/airflow-webserver -n airflow 8080:8080

# Trigger DAG
kubectl exec -n airflow deployment/airflow-webserver -- \
  airflow dags trigger my_dag

# Check logs
kubectl logs -n airflow -l component=scheduler -f

# List DAGs
kubectl exec -n airflow deployment/airflow-webserver -- \
  airflow dags list
```
