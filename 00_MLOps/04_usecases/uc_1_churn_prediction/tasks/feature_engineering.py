# uc_1_churn_prediction/tasks/feature_engineering.py
"""Feature engineering tasks using Feast."""

from prefect import task
import subprocess
from datetime import datetime
from pathlib import Path
import pandas as pd


@task(name="Prepare Feast Data")
def prepare_feast_data(file_path: str) -> str:
    """
    Prepare data for Feast by adding event_timestamp.

    Args:
        file_path: Path to raw parquet file

    Returns:
        str: Path to prepared parquet file
    """
    df = pd.read_parquet(file_path)

    # Add event_timestamp for Feast
    df['event_timestamp'] = pd.Timestamp.now()

    output_path = "data/customers.parquet"
    Path("data").mkdir(exist_ok=True)
    df.to_parquet(output_path, index=False)

    print(f"Prepared Feast data: {output_path}")
    return output_path


@task(name="Apply Feast Features")
def apply_feast_features() -> None:
    """Step 5a: Apply Feast feature definitions."""
    subprocess.run(
        ["feast", "apply"],
        cwd="feature_repo",
        check=True
    )
    print("Feast features applied")


@task(name="Materialize Feast Features")
def materialize_feast_features() -> str:
    """
    Step 5b: Materialize features to online store.

    Returns:
        str: Timestamp of materialization
    """
    end_date = datetime.now().strftime('%Y-%m-%dT%H:%M:%S')

    subprocess.run(
        ["feast", "materialize-incremental", end_date],
        cwd="feature_repo",
        check=True
    )

    print(f"Features materialized up to {end_date}")
    return end_date
