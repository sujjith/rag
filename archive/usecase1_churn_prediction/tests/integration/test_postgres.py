"""
Integration tests for PostgreSQL connectivity and data access.
"""

import pytest
import pandas as pd
from sqlalchemy import create_engine, text


class TestPostgresConnection:
    """Test PostgreSQL connectivity and data access."""
    
    @pytest.fixture
    def db_engine(self, postgres_config):
        """Create database engine."""
        cfg = postgres_config
        db_url = f"postgresql://{cfg['user']}:{cfg['password']}@{cfg['host']}:{cfg['port']}/{cfg['database']}"
        return create_engine(db_url)
    
    def test_connection(self, db_engine):
        """Test basic PostgreSQL connection."""
        with db_engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            assert result.fetchone()[0] == 1
        print("✓ PostgreSQL connection successful")
    
    def test_customers_table_exists(self, db_engine):
        """Test that customers table exists."""
        with db_engine.connect() as conn:
            result = conn.execute(text(
                "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'customers')"
            ))
            assert result.fetchone()[0] is True
        print("✓ Customers table exists")
    
    def test_customers_data_extraction(self, db_engine):
        """Test extracting data from customers table."""
        df = pd.read_sql("SELECT * FROM customers", db_engine)
        assert len(df) > 0, "No data in customers table"
        assert "customer_id" in df.columns
        assert "churn" in df.columns
        print(f"✓ Extracted {len(df)} customer records")
    
    def test_customers_schema(self, db_engine):
        """Test customers table has expected columns."""
        df = pd.read_sql("SELECT * FROM customers LIMIT 1", db_engine)
        expected_columns = [
            "customer_id", "age", "gender", "tenure_months",
            "total_purchases", "avg_order_value", 
            "days_since_last_purchase", "support_tickets_count", "churn"
        ]
        for col in expected_columns:
            assert col in df.columns, f"Missing column: {col}"
        print("✓ All expected columns present")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
