"""
Phase 10.1: Great Expectations Integration

Demonstrates:
- Data validation with Great Expectations
- Expectation suites
- Validation results handling
- Data quality gates
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.decorators import dag, task
from airflow.exceptions import AirflowException
import json


default_args = {
    "owner": "airflow",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


class DataValidator:
    """
    Simplified Great Expectations-style validator.
    In production, use actual Great Expectations library.
    """

    def __init__(self, data: list):
        self.data = data
        self.results = []

    def expect_column_to_exist(self, column: str) -> bool:
        """Check if column exists in data."""
        if not self.data:
            self.results.append({"expectation": f"column_exists_{column}", "success": False, "reason": "No data"})
            return False

        exists = column in self.data[0] if self.data else False
        self.results.append({
            "expectation": f"column_exists_{column}",
            "success": exists,
            "reason": None if exists else f"Column {column} not found"
        })
        return exists

    def expect_column_values_to_not_be_null(self, column: str) -> bool:
        """Check for null values in column."""
        null_count = sum(1 for row in self.data if row.get(column) is None)
        success = null_count == 0
        self.results.append({
            "expectation": f"no_nulls_{column}",
            "success": success,
            "details": {"null_count": null_count, "total_rows": len(self.data)},
            "reason": None if success else f"Found {null_count} null values"
        })
        return success

    def expect_column_values_to_be_between(self, column: str, min_val: float, max_val: float) -> bool:
        """Check if values are within range."""
        out_of_range = [
            row.get(column) for row in self.data
            if row.get(column) is not None and (row[column] < min_val or row[column] > max_val)
        ]
        success = len(out_of_range) == 0
        self.results.append({
            "expectation": f"values_between_{column}_{min_val}_{max_val}",
            "success": success,
            "details": {"out_of_range_count": len(out_of_range)},
            "reason": None if success else f"Found {len(out_of_range)} values outside range"
        })
        return success

    def expect_column_values_to_be_unique(self, column: str) -> bool:
        """Check for duplicate values."""
        values = [row.get(column) for row in self.data if row.get(column) is not None]
        duplicates = len(values) - len(set(values))
        success = duplicates == 0
        self.results.append({
            "expectation": f"unique_{column}",
            "success": success,
            "details": {"duplicate_count": duplicates},
            "reason": None if success else f"Found {duplicates} duplicate values"
        })
        return success

    def expect_table_row_count_to_be_between(self, min_count: int, max_count: int) -> bool:
        """Check row count is within range."""
        count = len(self.data)
        success = min_count <= count <= max_count
        self.results.append({
            "expectation": f"row_count_between_{min_count}_{max_count}",
            "success": success,
            "details": {"actual_count": count},
            "reason": None if success else f"Row count {count} outside range [{min_count}, {max_count}]"
        })
        return success

    def get_validation_results(self) -> dict:
        """Get all validation results."""
        return {
            "success": all(r["success"] for r in self.results),
            "results": self.results,
            "statistics": {
                "evaluated_expectations": len(self.results),
                "successful_expectations": sum(1 for r in self.results if r["success"]),
                "unsuccessful_expectations": sum(1 for r in self.results if not r["success"]),
            }
        }


@dag(
    dag_id="phase10_01_great_expectations",
    description="Data validation with Great Expectations patterns",
    default_args=default_args,
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False,
    tags=["phase10", "enterprise", "data-quality", "validation"],
    doc_md="""
    ## Great Expectations Integration

    **Prerequisites:**
    - Install: `pip install great-expectations`

    **Patterns Covered:**
    - Expectation suite definition
    - Column-level validations
    - Table-level validations
    - Validation results handling
    - Quality gates (fail pipeline on validation failure)

    **Common Expectations:**
    - `expect_column_to_exist`
    - `expect_column_values_to_not_be_null`
    - `expect_column_values_to_be_between`
    - `expect_column_values_to_be_unique`
    - `expect_table_row_count_to_be_between`
    """,
)
def great_expectations_validation():

    @task
    def extract_data():
        """Extract data for validation."""
        # Simulated data with some quality issues
        data = [
            {"id": 1, "name": "Alice", "age": 30, "email": "alice@example.com", "score": 85.5},
            {"id": 2, "name": "Bob", "age": 25, "email": "bob@example.com", "score": 92.0},
            {"id": 3, "name": None, "age": 35, "email": "charlie@example.com", "score": 78.5},  # Null name
            {"id": 4, "name": "Diana", "age": 150, "email": "diana@example.com", "score": 88.0},  # Invalid age
            {"id": 5, "name": "Eve", "age": 28, "email": "eve@example.com", "score": -5.0},  # Invalid score
            {"id": 5, "name": "Frank", "age": 32, "email": "frank@example.com", "score": 91.0},  # Duplicate ID
        ]

        return {"data": data, "source": "database", "extracted_at": datetime.now().isoformat()}

    @task
    def run_validation_suite(extracted: dict):
        """Run Great Expectations validation suite."""
        data = extracted["data"]
        validator = DataValidator(data)

        print("=" * 60)
        print("RUNNING VALIDATION SUITE")
        print("=" * 60)

        # Define expectations
        validator.expect_column_to_exist("id")
        validator.expect_column_to_exist("name")
        validator.expect_column_to_exist("age")
        validator.expect_column_to_exist("email")
        validator.expect_column_to_exist("score")

        validator.expect_column_values_to_not_be_null("id")
        validator.expect_column_values_to_not_be_null("name")
        validator.expect_column_values_to_not_be_null("email")

        validator.expect_column_values_to_be_unique("id")
        validator.expect_column_values_to_be_unique("email")

        validator.expect_column_values_to_be_between("age", 0, 120)
        validator.expect_column_values_to_be_between("score", 0, 100)

        validator.expect_table_row_count_to_be_between(1, 10000)

        results = validator.get_validation_results()

        # Print results
        print("\nVALIDATION RESULTS:")
        print("-" * 40)
        for r in results["results"]:
            status = "✅" if r["success"] else "❌"
            print(f"{status} {r['expectation']}")
            if not r["success"]:
                print(f"   Reason: {r['reason']}")

        print("\nSUMMARY:")
        print(f"  Total Expectations: {results['statistics']['evaluated_expectations']}")
        print(f"  Passed: {results['statistics']['successful_expectations']}")
        print(f"  Failed: {results['statistics']['unsuccessful_expectations']}")
        print(f"  Overall Success: {'✅' if results['success'] else '❌'}")

        return results

    @task
    def quality_gate(validation_results: dict):
        """
        Quality gate - fail pipeline if validation fails.
        Can be configured for soft or hard failure.
        """
        # Configuration
        fail_on_validation_error = True  # Set to False for soft failure
        critical_expectations = ["unique_id", "no_nulls_id"]

        failed = validation_results["statistics"]["unsuccessful_expectations"]
        total = validation_results["statistics"]["evaluated_expectations"]

        # Check critical expectations
        critical_failures = [
            r for r in validation_results["results"]
            if not r["success"] and any(crit in r["expectation"] for crit in critical_expectations)
        ]

        if critical_failures:
            error_msg = f"Critical validation failures: {[f['expectation'] for f in critical_failures]}"
            if fail_on_validation_error:
                raise AirflowException(error_msg)
            else:
                print(f"WARNING: {error_msg}")

        # Check overall threshold
        success_rate = (total - failed) / total if total > 0 else 0
        threshold = 0.8  # 80% expectations must pass

        if success_rate < threshold:
            error_msg = f"Validation success rate {success_rate:.1%} below threshold {threshold:.1%}"
            if fail_on_validation_error:
                raise AirflowException(error_msg)
            else:
                print(f"WARNING: {error_msg}")

        return {
            "gate_passed": success_rate >= threshold and not critical_failures,
            "success_rate": success_rate,
            "critical_failures": len(critical_failures),
        }

    @task
    def store_validation_results(validation_results: dict, gate_result: dict):
        """Store validation results for audit trail."""
        record = {
            "run_id": "{{ run_id }}",
            "dag_id": "phase10_01_great_expectations",
            "validation_time": datetime.now().isoformat(),
            "results": validation_results,
            "gate_passed": gate_result["gate_passed"],
        }

        print("Would store validation results:")
        print(json.dumps(record, indent=2, default=str)[:1000])

        # In production:
        # - Store in database
        # - Send to Great Expectations Data Docs
        # - Publish metrics

        return {"stored": True, "record_id": "val_123"}

    @task
    def proceed_with_processing(gate_result: dict, extracted: dict):
        """Process data if quality gate passed."""
        if not gate_result["gate_passed"]:
            print("Quality gate failed - skipping processing")
            return {"processed": False, "reason": "quality_gate_failed"}

        print(f"Processing {len(extracted['data'])} records...")
        return {"processed": True, "records": len(extracted["data"])}

    # DAG flow
    data = extract_data()
    validation = run_validation_suite(data)
    gate = quality_gate(validation)
    store_validation_results(validation, gate)
    proceed_with_processing(gate, data)


great_expectations_validation()
