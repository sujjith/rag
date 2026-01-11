
import great_expectations as gx
import pandas as pd
import sys
from great_expectations.core.expectation_suite import ExpectationSuite
from great_expectations.expectations import (
    ExpectColumnToExist,
    ExpectColumnValuesToNotBeNull,
    ExpectColumnValuesToBeInSet,
    ExpectColumnValuesToBeBetween
)

def validate_data(data_path):
    print(f"Validating data at {data_path}...")
    
    # Load data
    df = pd.read_csv(data_path)
    
    # Get context
    context = gx.get_context()
    
    # Create Expectation Suite
    suite_name = "customer_churn_suite"
    
    # Create or replace suite
    try:
        context.suites.delete(suite_name)
    except:
        pass
    
    suite = context.suites.add(ExpectationSuite(name=suite_name))

    # --- Define Expectations ---
    expectations = [
        # 1. Schema checks
        ExpectColumnToExist(column="customer_id"),
        ExpectColumnToExist(column="churn"),
        ExpectColumnValuesToNotBeNull(column="customer_id"),
        
        # 2. Type/Set checks
        ExpectColumnValuesToBeInSet(column="contract_type", value_set=['month-to-month', 'one-year', 'two-year']),
        ExpectColumnValuesToBeInSet(column="internet_service", value_set=['DSL', 'Fiber', 'None']),
        ExpectColumnValuesToBeInSet(column="churn", value_set=[0, 1]),
        
        # 3. Range checks
        ExpectColumnValuesToBeBetween(column="age", min_value=18, max_value=120),
        ExpectColumnValuesToBeBetween(column="tenure_months", min_value=0, max_value=120),
        ExpectColumnValuesToBeBetween(column="monthly_charges", min_value=0, max_value=1000)
    ]

    for expectation in expectations:
        suite.add_expectation(expectation)
    
    # Validate
    # Use ephemeral data source
    ds_name = "temp_pandas_datasource"
    try:
        context.data_sources.delete(ds_name)
    except:
        pass
        
    ds = context.data_sources.add_pandas(ds_name)
    asset = ds.add_dataframe_asset(name="dataframe_asset")
    batch_definition = asset.add_batch_definition_whole_dataframe("batch_definition")
    batch = batch_definition.get_batch(batch_parameters={"dataframe": df})
    
    result = batch.validate(suite)
    
    if result.success:
        print("✅ Data validation PASSED")
        return 0
    else:
        print("❌ Data validation FAILED")
        print(f"Success: {result.success}")
        # print details if needed
        return 1

if __name__ == "__main__":
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = "data/raw/customers.csv"
        
    sys.exit(validate_data(path))
