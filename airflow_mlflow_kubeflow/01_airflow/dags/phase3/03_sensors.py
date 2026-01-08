"""
Phase 3.3: Sensors DAG

Demonstrates:
- FileSensor
- PythonSensor
- ExternalTaskSensor
- TimeSensor

"""
from datetime import datetime, timedelta
from pathlib import Path
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.sensors.filesystem import FileSensor
from airflow.sensors.python import PythonSensor
from airflow.sensors.time_sensor import TimeSensor
from airflow.operators.empty import EmptyOperator

default_args = {
    "owner": "airflow",
    "retries": 1,
    "retry_delay": timedelta(minutes=1),
}


def check_condition(**kwargs):
    """Custom condition check for PythonSensor."""
    import random

    # Simulate checking an external condition
    # In real scenarios, this could check:
    # - API availability
    # - Database record existence
    # - External service status
    result = random.random() > 0.3  # 70% chance of success
    print(f"Condition check result: {result}")
    return result


def create_trigger_file(**kwargs):
    """Create a file to trigger FileSensor."""
    trigger_path = Path("/tmp/airflow_trigger_file.txt")
    trigger_path.write_text(f"Triggered at {datetime.now()}")
    print(f"Created trigger file: {trigger_path}")


def process_after_sensor(**kwargs):
    """Process after sensor detected condition."""
    print("Sensor condition met! Processing...")
    return "processed"


with DAG(
    dag_id="phase3_03_sensors",
    description="Sensor operators demonstrations",
    default_args=default_args,
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False,
    tags=["phase3", "operators", "sensors"],
    doc_md="""
    ## Sensors DAG

    Sensors are operators that wait for a certain condition to be met.

    **Common Sensors:**
    - `FileSensor` - Wait for file existence
    - `PythonSensor` - Wait for Python callable to return True
    - `TimeSensor` - Wait until specific time
    - `ExternalTaskSensor` - Wait for another DAG's task

    **Key Parameters:**
    - `poke_interval` - How often to check (seconds)
    - `timeout` - Maximum wait time (seconds)
    - `mode` - "poke" (blocking) or "reschedule" (non-blocking)
    """,
) as dag:

    start = EmptyOperator(task_id="start")

    # Create a trigger file for the FileSensor to detect
    create_file = PythonOperator(
        task_id="create_trigger_file",
        python_callable=create_trigger_file,
    )

    # FileSensor - wait for file
    file_sensor = FileSensor(
        task_id="wait_for_file",
        filepath="/tmp/airflow_trigger_file.txt",
        poke_interval=5,  # Check every 5 seconds
        timeout=60,  # Timeout after 60 seconds
        mode="poke",  # "poke" = blocking, "reschedule" = release worker
    )

    # PythonSensor - custom condition
    python_sensor = PythonSensor(
        task_id="wait_for_condition",
        python_callable=check_condition,
        poke_interval=3,
        timeout=60,
        mode="poke",
    )

    # TimeSensor - wait until specific time
    # Note: In practice, use TimeSensorAsync for better resource usage
    time_sensor = TimeSensor(
        task_id="wait_until_time",
        target_time=datetime.now().time(),  # Immediately pass (for demo)
        poke_interval=5,
        timeout=60,
    )

    # Process after all sensors
    process = PythonOperator(
        task_id="process_data",
        python_callable=process_after_sensor,
    )

    end = EmptyOperator(task_id="end")

    # Dependencies
    start >> create_file >> file_sensor
    start >> python_sensor
    start >> time_sensor

    [file_sensor, python_sensor, time_sensor] >> process >> end
