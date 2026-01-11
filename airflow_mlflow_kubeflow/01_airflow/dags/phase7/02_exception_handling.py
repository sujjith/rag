"""
Phase 7.2: Exception Handling Patterns

Demonstrates:
- Custom exception handling
- Error classification
- Graceful degradation
- Error recovery patterns
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.decorators import dag, task
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.exceptions import AirflowException, AirflowSkipException
from airflow.utils.trigger_rule import TriggerRule


# Custom exceptions for error classification
class TransientError(AirflowException):
    """Temporary error that may resolve on retry."""
    pass


class PermanentError(AirflowException):
    """Permanent error that should not retry."""
    pass


class DataValidationError(AirflowException):
    """Data quality issue detected."""
    pass


default_args = {
    "owner": "airflow",
    "retries": 2,
    "retry_delay": timedelta(seconds=10),
}


@dag(
    dag_id="phase7_02_exception_handling",
    description="Exception handling and error recovery patterns",
    default_args=default_args,
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False,
    tags=["phase7", "enterprise", "error-handling"],
    doc_md="""
    ## Exception Handling Patterns

    **Error Classification:**
    - TransientError: Temporary failures (network, timeout)
    - PermanentError: Unrecoverable errors (bad data, missing config)
    - DataValidationError: Data quality issues

    **Recovery Patterns:**
    1. Retry with backoff for transient errors
    2. Skip task for non-critical failures
    3. Fallback to default values
    4. Alert and continue for degraded mode

    **Best Practices:**
    - Classify errors appropriately
    - Use specific exception types
    - Implement graceful degradation
    - Log errors with context
    """,
)
def exception_handling_dag():

    @task
    def fetch_data():
        """Fetch data with error classification."""
        import random

        scenario = random.choice(["success", "transient", "permanent", "validation"])

        if scenario == "transient":
            raise TransientError("Network timeout - will retry")
        elif scenario == "permanent":
            raise PermanentError("API key invalid - cannot retry")
        elif scenario == "validation":
            raise DataValidationError("Data schema mismatch")

        return {"data": [1, 2, 3, 4, 5], "source": "api"}

    @task
    def fetch_data_with_fallback():
        """Fetch with fallback to cached data."""
        import random

        try:
            # Simulate API call
            if random.random() < 0.5:
                raise TransientError("API unavailable")

            return {"data": [10, 20, 30], "source": "live_api"}

        except TransientError as e:
            print(f"Primary source failed: {e}")
            print("Falling back to cached data...")

            # Return cached/default data instead of failing
            return {"data": [5, 10, 15], "source": "cache", "is_fallback": True}

    @task
    def process_with_skip_on_error(data: dict):
        """
        Process data, skip if non-critical error.
        Uses AirflowSkipException to skip without failing.
        """
        if data.get("is_fallback"):
            print("Using fallback data - skipping enrichment")
            raise AirflowSkipException("Skipping due to fallback data")

        return {"processed": sum(data["data"]), "source": data["source"]}

    @task
    def validate_data(data: dict):
        """Validate data with detailed error reporting."""
        errors = []

        if not data.get("data"):
            errors.append("Missing 'data' field")

        if not isinstance(data.get("data", []), list):
            errors.append("'data' must be a list")

        if len(data.get("data", [])) == 0:
            errors.append("'data' cannot be empty")

        if errors:
            error_msg = "; ".join(errors)
            raise DataValidationError(f"Validation failed: {error_msg}")

        print(f"Data validated successfully: {len(data['data'])} records")
        return {"validated": True, "record_count": len(data["data"])}

    @task(trigger_rule=TriggerRule.ALL_DONE)
    def error_handler(data: dict = None, validation: dict = None):
        """
        Handle errors from upstream tasks.
        Runs regardless of upstream status (ALL_DONE).
        """
        if data is None and validation is None:
            print("ERROR: All upstream tasks failed")
            # In production: send alert, log to error tracking
            return {"status": "all_failed", "action": "alert_sent"}

        if validation is None:
            print("WARNING: Validation task failed or was skipped")
            return {"status": "partial_success", "action": "logged"}

        print("SUCCESS: All tasks completed")
        return {"status": "success", "data": data, "validation": validation}

    @task
    def cleanup(result: dict):
        """Cleanup resources regardless of outcome."""
        print(f"Cleanup triggered with status: {result.get('status')}")

        # Close connections, release locks, etc.
        return "cleanup_complete"

    # DAG flow with error handling
    data = fetch_data_with_fallback()
    processed = process_with_skip_on_error(data)
    validated = validate_data(data)

    # Error handler runs regardless of upstream status
    result = error_handler(data, validated)
    cleanup(result)


exception_handling_dag()
