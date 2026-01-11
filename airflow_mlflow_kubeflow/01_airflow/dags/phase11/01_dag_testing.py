"""
Phase 11.1: DAG Testing Patterns

This file contains example DAGs and test patterns.
Tests should be in a separate tests/ directory.

Demonstrates:
- DAG validation tests
- Task unit tests
- Integration tests
- Test fixtures
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.decorators import dag, task
from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator


default_args = {
    "owner": "airflow",
    "retries": 1,
    "retry_delay": timedelta(minutes=1),
}


# ==================== Testable Functions ====================
# These functions are designed to be easily testable

def extract_data(source: str, date: str) -> dict:
    """
    Extract data from source.
    This function is easily testable in isolation.
    """
    if not source:
        raise ValueError("Source cannot be empty")
    if not date:
        raise ValueError("Date cannot be empty")

    # Simulate extraction
    return {
        "source": source,
        "date": date,
        "records": [
            {"id": 1, "value": 100},
            {"id": 2, "value": 200},
        ],
        "count": 2,
    }


def transform_data(data: dict) -> dict:
    """
    Transform extracted data.
    Pure function - same input always produces same output.
    """
    if not data or "records" not in data:
        raise ValueError("Invalid data format")

    transformed_records = []
    for record in data["records"]:
        transformed_records.append({
            "id": record["id"],
            "value": record["value"] * 2,  # Example transformation
            "processed_at": datetime.now().isoformat(),
        })

    return {
        "source": data["source"],
        "records": transformed_records,
        "count": len(transformed_records),
    }


def validate_output(data: dict) -> bool:
    """
    Validate transformed data.
    Returns True if valid, raises exception otherwise.
    """
    if not data:
        raise ValueError("Data cannot be empty")

    if data.get("count", 0) == 0:
        raise ValueError("No records to validate")

    for record in data.get("records", []):
        if record.get("value", 0) < 0:
            raise ValueError(f"Invalid value: {record['value']}")

    return True


# ==================== DAG Definition ====================

@dag(
    dag_id="phase11_01_testable_dag",
    description="A DAG designed for testability",
    default_args=default_args,
    start_date=datetime(2024, 1, 1),
    schedule="@daily",
    catchup=False,
    tags=["phase11", "enterprise", "testing"],
    doc_md="""
    ## Testable DAG

    This DAG demonstrates patterns for writing testable Airflow code.

    **Testing Patterns:**
    1. Separate business logic from Airflow operators
    2. Use pure functions where possible
    3. Design for dependency injection
    4. Keep tasks small and focused

    **Test Types:**
    - Unit tests for functions
    - DAG validation tests
    - Integration tests

    See `tests/test_phase11_dag.py` for example tests.
    """,
)
def testable_dag():

    @task
    def extract(**kwargs):
        """Extract task - wraps extract_data function."""
        ds = kwargs.get("ds", "2024-01-01")
        source = "database"
        return extract_data(source, ds)

    @task
    def transform(data: dict):
        """Transform task - wraps transform_data function."""
        return transform_data(data)

    @task
    def validate(data: dict):
        """Validate task - wraps validate_output function."""
        is_valid = validate_output(data)
        return {"valid": is_valid, "count": data["count"]}

    @task
    def load(validation_result: dict, data: dict):
        """Load task."""
        if not validation_result["valid"]:
            raise ValueError("Validation failed - cannot load")

        print(f"Loading {data['count']} records...")
        return {"loaded": True, "count": data["count"]}

    # DAG flow
    extracted = extract()
    transformed = transform(extracted)
    validated = validate(transformed)
    load(validated, transformed)


testable_dag()


# ==================== Example Test Code (for reference) ====================
"""
# tests/test_phase11_dag.py

import pytest
from datetime import datetime
from airflow.models import DagBag

# Import functions to test
from dags.phase11.01_dag_testing import (
    extract_data,
    transform_data,
    validate_output,
)


class TestDagValidation:
    '''Test DAG structure and validity.'''

    @pytest.fixture
    def dagbag(self):
        return DagBag(dag_folder='dags/', include_examples=False)

    def test_dag_loaded(self, dagbag):
        '''Test that DAG loads without errors.'''
        dag = dagbag.get_dag('phase11_01_testable_dag')
        assert dag is not None
        assert len(dagbag.import_errors) == 0

    def test_dag_has_correct_tasks(self, dagbag):
        '''Test that DAG has expected tasks.'''
        dag = dagbag.get_dag('phase11_01_testable_dag')
        task_ids = [task.task_id for task in dag.tasks]
        assert 'extract' in task_ids
        assert 'transform' in task_ids
        assert 'validate' in task_ids
        assert 'load' in task_ids

    def test_dag_has_correct_schedule(self, dagbag):
        '''Test DAG schedule.'''
        dag = dagbag.get_dag('phase11_01_testable_dag')
        assert dag.schedule_interval == '@daily'


class TestExtractData:
    '''Unit tests for extract_data function.'''

    def test_extract_success(self):
        result = extract_data('database', '2024-01-01')
        assert result['source'] == 'database'
        assert result['date'] == '2024-01-01'
        assert result['count'] == 2

    def test_extract_empty_source_raises(self):
        with pytest.raises(ValueError, match='Source cannot be empty'):
            extract_data('', '2024-01-01')

    def test_extract_empty_date_raises(self):
        with pytest.raises(ValueError, match='Date cannot be empty'):
            extract_data('database', '')


class TestTransformData:
    '''Unit tests for transform_data function.'''

    def test_transform_doubles_values(self):
        input_data = {
            'source': 'test',
            'records': [{'id': 1, 'value': 100}],
        }
        result = transform_data(input_data)
        assert result['records'][0]['value'] == 200

    def test_transform_invalid_data_raises(self):
        with pytest.raises(ValueError):
            transform_data({})


class TestValidateOutput:
    '''Unit tests for validate_output function.'''

    def test_validate_success(self):
        data = {'count': 1, 'records': [{'value': 100}]}
        assert validate_output(data) is True

    def test_validate_empty_raises(self):
        with pytest.raises(ValueError):
            validate_output({})

    def test_validate_negative_value_raises(self):
        data = {'count': 1, 'records': [{'value': -100}]}
        with pytest.raises(ValueError, match='Invalid value'):
            validate_output(data)


# Run with: pytest tests/test_phase11_dag.py -v
"""
