# uc_1_churn_prediction/flows/churn_pipeline.py
"""Main Prefect flow for UC1: Customer Churn Prediction Pipeline."""

from prefect import flow, get_run_logger

from uc_1_churn_prediction.tasks.data_ingestion import (
    extract_from_postgres,
    upload_to_minio
)
from uc_1_churn_prediction.tasks.data_versioning import version_with_dvc
from uc_1_churn_prediction.tasks.data_validation import validate_with_great_expectations
from uc_1_churn_prediction.tasks.feature_engineering import (
    prepare_feast_data,
    apply_feast_features,
    materialize_feast_features
)
from uc_1_churn_prediction.tasks.model_training import (
    train_model_local,
    get_latest_model_version
)
from uc_1_churn_prediction.tasks.model_serving import (
    deploy_to_kserve,
    wait_for_kserve_ready
)


@flow(name="UC1: Churn Prediction Pipeline", log_prints=True)
def churn_prediction_pipeline(
    skip_versioning: bool = False,
    skip_training: bool = False,
    skip_deployment: bool = False
) -> dict:
    """
    End-to-end customer churn prediction pipeline.

    Pipeline Steps:
    1. Extract data from PostgreSQL
    2. Upload to MinIO
    3. Version with DVC
    4. Validate with Great Expectations
    5. Engineer features with Feast
    6. Train model (logs to MLflow)
    7. Deploy to KServe

    Args:
        skip_versioning: Skip DVC versioning step
        skip_training: Skip model training step
        skip_deployment: Skip KServe deployment step

    Returns:
        dict: Pipeline results including paths and endpoints
    """
    logger = get_run_logger()
    results = {}

    # ─────────────────────────────────────────────────────────────
    # STEP 1: Data Ingestion
    # ─────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 1: Extracting data from PostgreSQL")
    logger.info("=" * 60)
    local_path = extract_from_postgres()
    results["local_path"] = local_path

    # ─────────────────────────────────────────────────────────────
    # STEP 2: Upload to Object Storage
    # ─────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 2: Uploading to MinIO")
    logger.info("=" * 60)
    s3_key = upload_to_minio(local_path)
    results["s3_key"] = s3_key

    # ─────────────────────────────────────────────────────────────
    # STEP 3: Data Versioning
    # ─────────────────────────────────────────────────────────────
    if not skip_versioning:
        logger.info("=" * 60)
        logger.info("STEP 3: Versioning with DVC")
        logger.info("=" * 60)
        dvc_file = version_with_dvc(local_path)
        results["dvc_file"] = dvc_file
    else:
        logger.info("STEP 3: Skipping DVC versioning")

    # ─────────────────────────────────────────────────────────────
    # STEP 4: Data Validation
    # ─────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 4: Validating with Great Expectations")
    logger.info("=" * 60)
    validation_passed = validate_with_great_expectations(local_path)
    results["validation_passed"] = validation_passed

    if not validation_passed:
        raise ValueError("Data validation failed! Pipeline aborted.")

    # ─────────────────────────────────────────────────────────────
    # STEP 5: Feature Engineering
    # ─────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 5: Engineering features with Feast")
    logger.info("=" * 60)

    # Prepare data for Feast
    feast_data_path = prepare_feast_data(local_path)

    # Apply feature definitions
    apply_feast_features()

    # Materialize to online store
    materialize_timestamp = materialize_feast_features()
    results["feast_materialized"] = materialize_timestamp

    # ─────────────────────────────────────────────────────────────
    # STEP 6: Model Training
    # ─────────────────────────────────────────────────────────────
    if not skip_training:
        logger.info("=" * 60)
        logger.info("STEP 6: Training model (MLflow tracking)")
        logger.info("=" * 60)
        run_id = train_model_local(local_path)
        results["mlflow_run_id"] = run_id

        # Get model info
        model_info = get_latest_model_version()
        results["model_info"] = model_info
    else:
        logger.info("STEP 6: Skipping model training")

    # ─────────────────────────────────────────────────────────────
    # STEP 7: Model Deployment
    # ─────────────────────────────────────────────────────────────
    if not skip_deployment and not skip_training:
        logger.info("=" * 60)
        logger.info("STEP 7: Deploying to KServe")
        logger.info("=" * 60)

        model_uri = results["model_info"]["source"]
        endpoint = deploy_to_kserve(model_uri)

        # Wait for ready
        wait_for_kserve_ready()
        results["endpoint"] = endpoint
    else:
        logger.info("STEP 7: Skipping KServe deployment")

    # ─────────────────────────────────────────────────────────────
    # COMPLETE
    # ─────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
    logger.info("=" * 60)
    logger.info(f"Results: {results}")

    return results


# ─────────────────────────────────────────────────────────────────
# Sub-flows for partial execution
# ─────────────────────────────────────────────────────────────────

@flow(name="UC1: Data Pipeline Only", log_prints=True)
def data_pipeline_only() -> dict:
    """Run only data ingestion, versioning, and validation steps."""
    return churn_prediction_pipeline(
        skip_training=True,
        skip_deployment=True
    )


@flow(name="UC1: Training Pipeline Only", log_prints=True)
def training_pipeline_only(data_path: str) -> dict:
    """Run only training step with existing data."""
    logger = get_run_logger()

    logger.info("Training model with existing data...")
    run_id = train_model_local(data_path)
    model_info = get_latest_model_version()

    return {
        "mlflow_run_id": run_id,
        "model_info": model_info
    }


# ─────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    churn_prediction_pipeline()
