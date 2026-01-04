# Phase 04: Data Engineering

**Duration**: 3 weeks | **Prerequisites**: Phase 03 completed

---

## Learning Objectives

By the end of this phase, you will:
- [ ] Process data efficiently with Polars
- [ ] Validate data quality with Great Expectations
- [ ] Build and serve features with Feast
- [ ] Create data transformations with dbt

---

## Week 1: Data Processing

### Day 1-3: Polars for Fast DataFrames

```bash
uv add polars
```

```python
import polars as pl

# Read data (much faster than pandas)
df = pl.read_csv("data.csv")

# Lazy evaluation for optimization
lazy_df = pl.scan_csv("large_data.csv")

# Transformations with method chaining
result = (
    lazy_df
    .filter(pl.col("age") > 25)
    .group_by("category")
    .agg([
        pl.col("value").mean().alias("avg_value"),
        pl.col("value").sum().alias("total_value"),
        pl.count().alias("count")
    ])
    .sort("avg_value", descending=True)
    .collect()  # Execute lazy query
)

# Window functions
df = df.with_columns([
    pl.col("value").rolling_mean(window_size=7).alias("rolling_avg"),
    pl.col("value").rank().over("category").alias("rank_in_category")
])
```

### Day 4-5: PySpark Basics

```bash
uv add pyspark
```

```python
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# Create session
spark = SparkSession.builder \
    .appName("MLDataProcessing") \
    .getOrCreate()

# Read data
df = spark.read.csv("data.csv", header=True, inferSchema=True)

# Transformations
result = (
    df
    .filter(F.col("age") > 25)
    .groupBy("category")
    .agg(
        F.avg("value").alias("avg_value"),
        F.count("*").alias("count")
    )
)

# ML feature engineering
from pyspark.ml.feature import VectorAssembler, StandardScaler

assembler = VectorAssembler(inputCols=["feat1", "feat2"], outputCol="features")
scaler = StandardScaler(inputCol="features", outputCol="scaled_features")

# Save as Parquet
result.write.parquet("output/result.parquet")
```

---

## Week 2: Data Quality

### Day 8-10: Great Expectations

```bash
uv add great_expectations
```

```bash
# Initialize GX project
uv run great_expectations init
```

```python
import great_expectations as gx

# Create context
context = gx.get_context()

# Add data source
datasource = context.sources.add_pandas("my_datasource")
data_asset = datasource.add_csv_asset("my_data", filepath_or_buffer="data.csv")

# Create expectations
batch_request = data_asset.build_batch_request()
validator = context.get_validator(batch_request=batch_request)

# Add expectations
validator.expect_column_to_exist("user_id")
validator.expect_column_values_to_not_be_null("user_id")
validator.expect_column_values_to_be_unique("user_id")
validator.expect_column_values_to_be_between("age", min_value=0, max_value=120)
validator.expect_column_values_to_be_in_set("status", ["active", "inactive"])

# Save expectation suite
validator.save_expectation_suite()

# Run validation
results = validator.validate()
print(f"Success: {results.success}")
```

### Day 11-12: Pandera for DataFrame Validation

```bash
uv add pandera
```

```python
import pandera as pa
from pandera import Column, Check
import pandas as pd

# Define schema
schema = pa.DataFrameSchema({
    "user_id": Column(int, Check.greater_than(0), unique=True),
    "email": Column(str, Check.str_matches(r"^[\w\.-]+@[\w\.-]+\.\w+$")),
    "age": Column(int, Check.in_range(0, 120)),
    "score": Column(float, Check.in_range(0, 1)),
    "category": Column(str, Check.isin(["A", "B", "C"])),
})

# Validate
@pa.check_input(schema)
def process_data(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("category").mean()

# Use as decorator
validated_df = schema.validate(df)
```

---

## Week 3: Feature Store with Feast

### Day 15-17: Feast Setup

```bash
uv add feast
```

```bash
# Initialize feature repo
mkdir feature_repo && cd feature_repo
uv run feast init
```

```python
# feature_repo/features.py
from feast import Entity, Feature, FeatureView, FileSource, ValueType
from datetime import timedelta

# Define entity
user = Entity(
    name="user_id",
    value_type=ValueType.INT64,
    description="User identifier"
)

# Define data source
user_stats_source = FileSource(
    path="data/user_stats.parquet",
    timestamp_field="event_timestamp",
)

# Define feature view
user_stats_fv = FeatureView(
    name="user_stats",
    entities=[user],
    ttl=timedelta(days=1),
    schema=[
        Feature(name="total_purchases", dtype=ValueType.INT64),
        Feature(name="avg_purchase_value", dtype=ValueType.FLOAT),
        Feature(name="days_since_last_purchase", dtype=ValueType.INT64),
    ],
    source=user_stats_source,
)
```

```bash
# Apply feature definitions
uv run feast apply
```

### Day 18-19: Serving Features

```python
from feast import FeatureStore
from datetime import datetime

store = FeatureStore(repo_path="feature_repo")

# Get features for training (point-in-time join)
training_df = store.get_historical_features(
    entity_df=entity_df,  # DataFrame with user_id and timestamps
    features=[
        "user_stats:total_purchases",
        "user_stats:avg_purchase_value",
    ],
).to_df()

# Online serving (real-time)
store.materialize_incremental(end_date=datetime.now())

feature_vector = store.get_online_features(
    features=[
        "user_stats:total_purchases",
        "user_stats:avg_purchase_value",
    ],
    entity_rows=[{"user_id": 12345}]
).to_dict()
```

### Day 20-21: dbt for Transformations

```bash
uv add dbt-core dbt-postgres  # or dbt-duckdb for local
```

```bash
# Initialize project
uv run dbt init my_project
```

```sql
-- models/staging/stg_users.sql
select
    user_id,
    email,
    created_at,
    case 
        when age < 25 then 'young'
        when age < 40 then 'middle'
        else 'senior'
    end as age_group
from {{ source('raw', 'users') }}
where user_id is not null
```

```sql
-- models/marts/user_features.sql
select
    u.user_id,
    u.age_group,
    count(o.order_id) as total_orders,
    sum(o.amount) as total_spent,
    avg(o.amount) as avg_order_value
from {{ ref('stg_users') }} u
left join {{ ref('stg_orders') }} o on u.user_id = o.user_id
group by u.user_id, u.age_group
```

```bash
# Run dbt
uv run dbt run
uv run dbt test
uv run dbt docs generate
uv run dbt docs serve
```

---

## Milestone Checklist

- [ ] Processed data with Polars
- [ ] Created PySpark job
- [ ] Built Great Expectations suite
- [ ] Used Pandera for validation
- [ ] Feast feature store running
- [ ] Served features online
- [ ] dbt models created

---

**Next Phase**: [Phase 05 - Monitoring & Observability](./phase_05_monitoring.md)
