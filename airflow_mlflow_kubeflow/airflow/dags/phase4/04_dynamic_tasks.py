"""
Phase 4.4: Dynamic Task Mapping DAG

Demonstrates:
- Dynamic task mapping with expand()
- Partial() for fixed parameters
- Map over multiple inputs
- Reduce patterns

"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.decorators import dag, task

default_args = {
    "owner": "airflow",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


@dag(
    dag_id="phase4_04_dynamic_tasks",
    description="Dynamic task mapping demonstration",
    default_args=default_args,
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False,
    tags=["phase4", "dynamic", "mapping"],
    doc_md="""
    ## Dynamic Task Mapping DAG

    Dynamic task mapping allows creating tasks at runtime based on data.

    **Key Concepts:**
    - `expand()` - Map over iterable to create tasks
    - `partial()` - Fix some parameters
    - Works with TaskFlow and traditional operators
    - Results are collected automatically
    """,
)
def dynamic_tasks_demo():
    """Demonstrate dynamic task mapping."""

    @task
    def get_data_sources():
        """Return list of data sources to process."""
        sources = [
            {"id": 1, "name": "source_a", "size": 100},
            {"id": 2, "name": "source_b", "size": 250},
            {"id": 3, "name": "source_c", "size": 50},
            {"id": 4, "name": "source_d", "size": 175},
        ]
        print(f"Found {len(sources)} sources to process")
        return sources

    @task
    def process_source(source: dict) -> dict:
        """Process a single data source (dynamically mapped)."""
        print(f"Processing {source['name']} with {source['size']} records...")
        # Simulate processing
        result = {
            "source_id": source["id"],
            "source_name": source["name"],
            "processed_records": source["size"],
            "success": True,
        }
        print(f"Completed: {result}")
        return result

    @task
    def aggregate_results(results: list) -> dict:
        """Aggregate all processed results."""
        total_records = sum(r["processed_records"] for r in results)
        successful = sum(1 for r in results if r["success"])

        summary = {
            "total_sources": len(results),
            "successful_sources": successful,
            "total_records_processed": total_records,
        }
        print(f"Aggregation complete: {summary}")
        return summary

    # Get sources dynamically
    sources = get_data_sources()

    # Map process_source over each source - creates N tasks at runtime!
    processed = process_source.expand(source=sources)

    # Aggregate all results
    aggregate_results(processed)


dynamic_tasks_demo()


# More advanced example with partial
@dag(
    dag_id="phase4_04b_advanced_mapping",
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False,
    tags=["phase4", "dynamic", "advanced"],
)
def advanced_mapping_demo():
    """Advanced dynamic mapping patterns."""

    @task
    def get_items():
        return [1, 2, 3, 4, 5]

    @task
    def get_multipliers():
        return [10, 100]

    @task
    def process_with_config(item: int, multiplier: int, prefix: str) -> str:
        """Process with fixed and dynamic parameters."""
        result = item * multiplier
        output = f"{prefix}: {item} x {multiplier} = {result}"
        print(output)
        return output

    @task
    def summarize(all_results: list):
        """Summarize all combinations."""
        print(f"Processed {len(all_results)} combinations")
        for r in all_results:
            print(f"  - {r}")

    items = get_items()
    multipliers = get_multipliers()

    # Use partial to fix the prefix, expand over items and multipliers
    # This creates tasks for each combination!
    results = process_with_config.partial(prefix="Result").expand(
        item=items,
        multiplier=multipliers,
    )

    summarize(results)


advanced_mapping_demo()


# Zip pattern example
@dag(
    dag_id="phase4_04c_zip_mapping",
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False,
    tags=["phase4", "dynamic", "zip"],
)
def zip_mapping_demo():
    """Demonstrate zip pattern for parallel lists."""

    @task
    def get_files():
        return ["file1.csv", "file2.csv", "file3.csv"]

    @task
    def get_destinations():
        return ["bucket_a", "bucket_b", "bucket_c"]

    @task
    def transfer(file: str, destination: str):
        """Transfer file to destination (1:1 mapping)."""
        print(f"Transferring {file} to {destination}")
        return {"file": file, "destination": destination, "status": "success"}

    @task
    def report(transfers: list):
        print(f"Completed {len(transfers)} transfers")

    files = get_files()
    destinations = get_destinations()

    # expand_kwargs with zip for 1:1 mapping (not cartesian product)
    from airflow.models.xcom_arg import XComArg

    # Create paired transfers
    transfers = transfer.expand(file=files, destination=destinations)
    report(transfers)


zip_mapping_demo()
