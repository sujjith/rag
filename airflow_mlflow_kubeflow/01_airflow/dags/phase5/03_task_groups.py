"""
Phase 5.3: Task Groups DAG

Demonstrates:
- TaskGroup for visual organization
- Nested task groups
- Reusable task group patterns

"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.decorators import dag, task, task_group
from airflow.operators.empty import EmptyOperator
from airflow.utils.task_group import TaskGroup

default_args = {
    "owner": "airflow",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


# Classic context manager approach
with DAG(
    dag_id="phase5_03_task_groups",
    description="Task Groups demonstration",
    default_args=default_args,
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False,
    tags=["phase5", "task_groups"],
    doc_md="""
    ## Task Groups DAG

    Task Groups organize tasks visually in the UI:
    - Collapsible groups for complex DAGs
    - Logical organization
    - Reusable patterns

    **Benefits:**
    - Cleaner DAG visualization
    - Modular DAG design
    - Easier navigation
    """,
) as dag:

    start = EmptyOperator(task_id="start")

    # Task Group for data extraction
    with TaskGroup(group_id="extract") as extract_group:
        extract_users = EmptyOperator(task_id="extract_users")
        extract_orders = EmptyOperator(task_id="extract_orders")
        extract_products = EmptyOperator(task_id="extract_products")

    # Task Group for transformation
    with TaskGroup(group_id="transform") as transform_group:
        clean_data = EmptyOperator(task_id="clean_data")
        normalize = EmptyOperator(task_id="normalize")
        validate = EmptyOperator(task_id="validate")

        # Order within group
        clean_data >> normalize >> validate

    # Task Group for loading
    with TaskGroup(group_id="load") as load_group:
        load_warehouse = EmptyOperator(task_id="load_warehouse")
        load_analytics = EmptyOperator(task_id="load_analytics")

    end = EmptyOperator(task_id="end")

    # Connect groups
    start >> extract_group >> transform_group >> load_group >> end


# TaskFlow decorator approach
@dag(
    dag_id="phase5_03b_taskflow_groups",
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False,
    tags=["phase5", "task_groups", "taskflow"],
)
def taskflow_groups_demo():
    """TaskFlow API with task groups."""

    @task
    def start():
        return "started"

    @task_group
    def extract_data():
        """Extract data from multiple sources."""

        @task
        def extract_api():
            return {"source": "api", "records": 100}

        @task
        def extract_db():
            return {"source": "database", "records": 500}

        @task
        def extract_files():
            return {"source": "files", "records": 250}

        # Return all extractions
        return {
            "api": extract_api(),
            "db": extract_db(),
            "files": extract_files(),
        }

    @task_group
    def transform_data(extracted: dict):
        """Transform extracted data."""

        @task
        def merge_sources(data: dict):
            total = sum(d["records"] for d in data.values())
            return {"merged_records": total}

        @task
        def apply_business_rules(merged: dict):
            return {"processed": merged["merged_records"], "valid": True}

        merged = merge_sources(extracted)
        return apply_business_rules(merged)

    @task_group
    def load_data(transformed: dict):
        """Load transformed data."""

        @task
        def load_to_warehouse(data: dict):
            print(f"Loading {data['processed']} records to warehouse")
            return "warehouse_loaded"

        @task
        def load_to_cache(data: dict):
            print(f"Caching {data['processed']} records")
            return "cache_loaded"

        return {
            "warehouse": load_to_warehouse(transformed),
            "cache": load_to_cache(transformed),
        }

    @task
    def complete(results: dict):
        print(f"Pipeline complete: {results}")

    # Wire it all together
    s = start()
    extracted = extract_data()
    transformed = transform_data(extracted)
    loaded = load_data(transformed)
    complete(loaded)


taskflow_groups_demo()


# Nested task groups example
@dag(
    dag_id="phase5_03c_nested_groups",
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False,
    tags=["phase5", "task_groups", "nested"],
)
def nested_groups_demo():
    """Demonstrate nested task groups."""

    @task
    def init():
        return "initialized"

    @task_group
    def region_processing():
        """Process data by region."""

        @task_group
        def us_region():
            @task
            def extract_us():
                return "us_data"

            @task
            def transform_us(data):
                return f"transformed_{data}"

            return transform_us(extract_us())

        @task_group
        def eu_region():
            @task
            def extract_eu():
                return "eu_data"

            @task
            def transform_eu(data):
                return f"transformed_{data}"

            return transform_eu(extract_eu())

        return {"us": us_region(), "eu": eu_region()}

    @task
    def aggregate(region_data: dict):
        print(f"Aggregating: {region_data}")
        return "aggregated"

    data = init()
    regions = region_processing()
    aggregate(regions)


nested_groups_demo()
