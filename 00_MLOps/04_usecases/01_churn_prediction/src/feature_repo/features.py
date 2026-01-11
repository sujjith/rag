from datetime import timedelta
from feast import Entity, FeatureView, Field, FileSource, ValueType
from feast.types import Float32, Int64, String, Int32
import os

# Define the source of feature data
customer_stats_source = FileSource(
    name="customer_stats_source",
    path="s3://ml-data/customers.csv",
    s3_endpoint_override="http://minio.minio.svc.cluster.local:9000",
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp",
)

# Define an entity
customer = Entity(
    name="customer_id", 
    value_type=ValueType.INT64, 
    description="customer identifier"
)

# Define a Feature View
customer_features = FeatureView(
    name="customer_features",
    entities=[customer],
    ttl=timedelta(days=3650),
    schema=[
        Field(name="age", dtype=Int64),
        Field(name="tenure_months", dtype=Int64),
        Field(name="monthly_charges", dtype=Float32),
        Field(name="total_charges", dtype=Float32),
        Field(name="num_support_tickets", dtype=Int64),
        Field(name="num_products", dtype=Int64),
        Field(name="payment_delay_days", dtype=Int64),
        Field(name="contract_type", dtype=String),
        Field(name="internet_service", dtype=String),
        Field(name="churn", dtype=Int64),
    ],
    source=customer_stats_source,
    online=True,
)
