"""
Phase 9.1: Database Integration

Demonstrates:
- PostgreSQL/MySQL connections
- SQL execution patterns
- Data extraction and loading
- Transaction management
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.decorators import dag, task
from airflow.providers.postgres.operators.postgres import PostgresOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.operators.empty import EmptyOperator
from airflow.utils.trigger_rule import TriggerRule


default_args = {
    "owner": "airflow",
    "retries": 2,
    "retry_delay": timedelta(minutes=1),
}


@dag(
    dag_id="phase9_01_database_integration",
    description="Database integration patterns",
    default_args=default_args,
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False,
    tags=["phase9", "enterprise", "database", "integration"],
    doc_md="""
    ## Database Integration

    **Prerequisites:**
    1. Create PostgreSQL connection in Airflow:
       - Connection ID: `postgres_default`
       - Connection Type: Postgres
       - Host: your-db-host
       - Schema: your_database
       - Login: username
       - Password: password
       - Port: 5432

    **Patterns Covered:**
    - Direct SQL execution
    - Data extraction with hooks
    - Bulk inserts
    - Transaction management
    - Error handling
    """,
)
def database_integration():

    @task
    def create_staging_table():
        """Create staging table for data loading."""
        # In production, use PostgresOperator or Hook
        create_sql = """
        CREATE TABLE IF NOT EXISTS staging_data (
            id SERIAL PRIMARY KEY,
            name VARCHAR(100),
            value DECIMAL(10, 2),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            batch_id VARCHAR(50)
        );
        """

        print(f"Would execute:\n{create_sql}")

        # In production:
        # hook = PostgresHook(postgres_conn_id="postgres_default")
        # hook.run(create_sql)

        return {"table": "staging_data", "created": True}

    @task
    def extract_from_source():
        """Extract data from source database."""
        # Simulated extraction
        query = """
        SELECT id, name, value, created_at
        FROM source_table
        WHERE created_at >= '{{ ds }}'
        AND created_at < '{{ next_ds }}'
        """

        print(f"Would execute:\n{query}")

        # In production:
        # hook = PostgresHook(postgres_conn_id="source_db")
        # records = hook.get_records(query)

        # Simulated data
        records = [
            (1, "Item A", 100.50, "2024-01-01"),
            (2, "Item B", 200.75, "2024-01-01"),
            (3, "Item C", 150.25, "2024-01-01"),
        ]

        print(f"Extracted {len(records)} records")
        return {"records": records, "count": len(records)}

    @task
    def transform_data(extracted: dict):
        """Transform extracted data."""
        records = extracted["records"]

        transformed = []
        for record in records:
            transformed.append({
                "id": record[0],
                "name": record[1].upper(),  # Example transformation
                "value": float(record[2]) * 1.1,  # Apply markup
                "created_at": record[3],
            })

        print(f"Transformed {len(transformed)} records")
        return {"records": transformed, "count": len(transformed)}

    @task
    def load_to_target(transformed: dict):
        """Load data to target database."""
        records = transformed["records"]
        batch_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        insert_sql = """
        INSERT INTO staging_data (name, value, batch_id)
        VALUES (%s, %s, %s)
        """

        print(f"Would insert {len(records)} records with batch_id: {batch_id}")

        # In production:
        # hook = PostgresHook(postgres_conn_id="postgres_default")
        # for record in records:
        #     hook.run(insert_sql, parameters=(
        #         record["name"],
        #         record["value"],
        #         batch_id
        #     ))

        return {"loaded": len(records), "batch_id": batch_id}

    @task
    def execute_stored_procedure(load_result: dict):
        """Execute stored procedure for final processing."""
        batch_id = load_result["batch_id"]

        proc_sql = f"""
        CALL process_staging_data('{batch_id}');
        """

        print(f"Would execute:\n{proc_sql}")

        # In production:
        # hook = PostgresHook(postgres_conn_id="postgres_default")
        # hook.run(proc_sql)

        return {"procedure": "process_staging_data", "executed": True}

    @task
    def run_quality_checks(load_result: dict):
        """Run data quality checks on loaded data."""
        batch_id = load_result["batch_id"]

        checks = [
            {
                "name": "row_count",
                "query": f"SELECT COUNT(*) FROM staging_data WHERE batch_id = '{batch_id}'",
                "expected": load_result["loaded"],
            },
            {
                "name": "null_check",
                "query": f"SELECT COUNT(*) FROM staging_data WHERE name IS NULL AND batch_id = '{batch_id}'",
                "expected": 0,
            },
            {
                "name": "value_range",
                "query": f"SELECT COUNT(*) FROM staging_data WHERE value < 0 AND batch_id = '{batch_id}'",
                "expected": 0,
            },
        ]

        results = []
        for check in checks:
            print(f"Running check: {check['name']}")
            print(f"Query: {check['query']}")

            # In production:
            # hook = PostgresHook(postgres_conn_id="postgres_default")
            # result = hook.get_first(check['query'])[0]
            # passed = result == check['expected']

            results.append({
                "check": check["name"],
                "passed": True,  # Simulated
            })

        return {"checks": results, "all_passed": all(r["passed"] for r in results)}

    @task
    def cleanup_staging(quality_result: dict, load_result: dict):
        """Clean up staging data after successful load."""
        if not quality_result["all_passed"]:
            print("Quality checks failed - keeping staging data for investigation")
            return {"cleaned": False, "reason": "quality_check_failed"}

        batch_id = load_result["batch_id"]
        cleanup_sql = f"""
        DELETE FROM staging_data
        WHERE batch_id = '{batch_id}'
        AND created_at < NOW() - INTERVAL '7 days'
        """

        print(f"Would execute:\n{cleanup_sql}")

        # In production:
        # hook = PostgresHook(postgres_conn_id="postgres_default")
        # hook.run(cleanup_sql)

        return {"cleaned": True, "batch_id": batch_id}

    # DAG flow
    table = create_staging_table()
    extracted = extract_from_source()
    transformed = transform_data(extracted)
    loaded = load_to_target(transformed)
    proc = execute_stored_procedure(loaded)
    quality = run_quality_checks(loaded)
    cleanup_staging(quality, loaded)

    table >> extracted


database_integration()
