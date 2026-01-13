"""
Pytest configuration and shared fixtures.
"""

import pytest


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "integration: mark test as integration test (requires infrastructure)"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )


@pytest.fixture(scope="session")
def postgres_config():
    """PostgreSQL connection configuration."""
    return {
        "host": "localhost",
        "port": 30432,
        "user": "postgres",
        "password": "postgres123",
        "database": "customers",
    }


@pytest.fixture(scope="session")
def minio_config():
    """MinIO connection configuration."""
    return {
        "endpoint_url": "http://localhost:30900",
        "access_key": "minioadmin",
        "secret_key": "minioadmin123",
        "bucket": "dvc-storage",
    }
