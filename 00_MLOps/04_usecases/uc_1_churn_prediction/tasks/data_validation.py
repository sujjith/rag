# uc_1_churn_prediction/tasks/data_validation.py
"""Data validation tasks using Great Expectations."""

from prefect import task
import great_expectations as gx
import pandas as pd


@task(name="Validate with Great Expectations")
def validate_with_great_expectations(file_path: str) -> bool:
    """
    Step 4: Validate data quality with Great Expectations.

    Args:
        file_path: Path to parquet file to validate

    Returns:
        bool: True if validation passes

    Raises:
        ValueError: If validation fails
    """
    df = pd.read_parquet(file_path)

    # Get GX context
    context = gx.get_context()

    # Create validator from dataframe
    validator = context.sources.pandas_default.read_dataframe(df)

    # Define expectations
    validator.expect_column_to_exist("customer_id")
    validator.expect_column_values_to_not_be_null("customer_id")
    validator.expect_column_values_to_be_unique("customer_id")

    validator.expect_column_to_exist("age")
    validator.expect_column_values_to_be_between("age", min_value=18, max_value=120)

    validator.expect_column_to_exist("churn")
    validator.expect_column_values_to_be_in_set("churn", [0, 1])

    validator.expect_column_values_to_be_between("tenure_months", min_value=0)
    validator.expect_column_values_to_be_between("total_purchases", min_value=0)
    validator.expect_column_values_to_be_between("avg_order_value", min_value=0)

    # Run validation
    results = validator.validate()

    if not results.success:
        failed = [r for r in results.results if not r.success]
        error_msg = f"Data validation failed! {len(failed)} expectations failed."
        print(error_msg)
        for f in failed:
            print(f"  - {f.expectation_config.expectation_type}")
        raise ValueError(error_msg)

    print(f"Data validation passed! {len(results.results)} expectations checked.")
    return True
