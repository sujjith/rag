# feature_repo/features.py
"""Feast feature definitions for churn prediction."""

from datetime import timedelta
from feast import Entity, FeatureView, FileSource, Field
from feast.types import Float32, Int32, String

# Define entity
customer = Entity(
    name="customer_id",
    join_keys=["customer_id"],
    description="Customer identifier"
)

# Define data source
customer_source = FileSource(
    path="data/customers.parquet",
    timestamp_field="event_timestamp"
)

# Define feature view
customer_features = FeatureView(
    name="customer_features",
    entities=[customer],
    ttl=timedelta(days=1),
    schema=[
        Field(name="age", dtype=Int32),
        Field(name="gender", dtype=String),
        Field(name="tenure_months", dtype=Int32),
        Field(name="total_purchases", dtype=Int32),
        Field(name="avg_order_value", dtype=Float32),
        Field(name="days_since_last_purchase", dtype=Int32),
        Field(name="support_tickets_count", dtype=Int32),
    ],
    source=customer_source,
)
