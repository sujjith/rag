"""
Phase 6.2: Complete ML Pipeline DAG

Demonstrates:
- End-to-end ML pipeline
- Data validation
- Feature engineering
- Model training with MLflow
- Model evaluation and deployment
- Notification on completion

"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.decorators import dag, task, task_group
from airflow.operators.empty import EmptyOperator
from airflow.utils.trigger_rule import TriggerRule

default_args = {
    "owner": "airflow",
    "retries": 2,
    "retry_delay": timedelta(minutes=2),
}


def notify_success(context):
    """Notify on pipeline success."""
    print("=" * 50)
    print("ML PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"DAG: {context['dag'].dag_id}")
    print(f"Execution Date: {context['ds']}")
    print("=" * 50)


def notify_failure(context):
    """Notify on pipeline failure."""
    print("=" * 50)
    print("ML PIPELINE FAILED!")
    print(f"DAG: {context['dag'].dag_id}")
    print(f"Failed Task: {context['task_instance'].task_id}")
    print(f"Exception: {context.get('exception')}")
    print("=" * 50)


@dag(
    dag_id="phase6_02_complete_ml_pipeline",
    description="Complete ML pipeline with all stages",
    default_args=default_args,
    start_date=datetime(2024, 1, 1),
    schedule="@daily",  # Run daily
    catchup=False,
    max_active_runs=1,
    on_success_callback=notify_success,
    on_failure_callback=notify_failure,
    tags=["phase6", "ml", "production"],
    doc_md="""
    ## Complete ML Pipeline

    Production-ready ML pipeline with:
    1. **Data Ingestion** - Load data from sources
    2. **Data Validation** - Check data quality
    3. **Feature Engineering** - Transform features
    4. **Model Training** - Train with MLflow tracking
    5. **Model Evaluation** - Validate model performance
    6. **Model Deployment** - Deploy to staging/production
    """,
)
def complete_ml_pipeline():
    """Complete ML Pipeline."""

    @task
    def start_pipeline():
        """Initialize pipeline run."""
        import uuid

        run_id = str(uuid.uuid4())[:8]
        print(f"Starting pipeline run: {run_id}")
        return {"pipeline_run_id": run_id, "start_time": str(datetime.now())}

    # ==================== DATA INGESTION ====================
    @task_group
    def data_ingestion():
        """Ingest data from multiple sources."""

        @task
        def ingest_from_database():
            """Ingest data from database."""
            print("Ingesting from database...")
            import time
            time.sleep(1)
            return {"source": "database", "records": 5000}

        @task
        def ingest_from_api():
            """Ingest data from API."""
            print("Ingesting from API...")
            import time
            time.sleep(1)
            return {"source": "api", "records": 1000}

        @task
        def ingest_from_files():
            """Ingest data from files."""
            print("Ingesting from files...")
            import time
            time.sleep(1)
            return {"source": "files", "records": 2000}

        @task
        def merge_data(db_data, api_data, file_data):
            """Merge all data sources."""
            total_records = db_data["records"] + api_data["records"] + file_data["records"]
            print(f"Merged {total_records} total records")
            return {
                "total_records": total_records,
                "sources": [db_data["source"], api_data["source"], file_data["source"]],
            }

        db = ingest_from_database()
        api = ingest_from_api()
        files = ingest_from_files()
        return merge_data(db, api, files)

    # ==================== DATA VALIDATION ====================
    @task_group
    def data_validation(data_info: dict):
        """Validate data quality."""

        @task
        def check_completeness(data: dict):
            """Check for missing values."""
            print(f"Checking completeness for {data['total_records']} records...")
            # Simulate check
            missing_rate = 0.02  # 2% missing
            return {
                "check": "completeness",
                "passed": missing_rate < 0.05,
                "missing_rate": missing_rate,
            }

        @task
        def check_schema(data: dict):
            """Validate data schema."""
            print("Validating schema...")
            return {"check": "schema", "passed": True}

        @task
        def check_statistics(data: dict):
            """Check data statistics for anomalies."""
            print("Checking statistics...")
            return {"check": "statistics", "passed": True, "anomalies": 0}

        @task
        def validation_summary(completeness, schema, stats):
            """Summarize validation results."""
            all_passed = all([completeness["passed"], schema["passed"], stats["passed"]])
            print(f"Validation {'PASSED' if all_passed else 'FAILED'}")
            return {"all_passed": all_passed, "checks": [completeness, schema, stats]}

        comp = check_completeness(data_info)
        schema = check_schema(data_info)
        stats = check_statistics(data_info)
        return validation_summary(comp, schema, stats)

    # ==================== FEATURE ENGINEERING ====================
    @task_group
    def feature_engineering(validation_result: dict):
        """Engineer features for training."""

        @task
        def create_numerical_features():
            """Create numerical features."""
            print("Creating numerical features...")
            return {"features": ["age_scaled", "income_normalized", "score_binned"], "count": 3}

        @task
        def create_categorical_features():
            """Create categorical features."""
            print("Creating categorical features...")
            return {"features": ["city_encoded", "category_onehot"], "count": 2}

        @task
        def create_derived_features():
            """Create derived/interaction features."""
            print("Creating derived features...")
            return {"features": ["age_income_ratio", "score_percentile"], "count": 2}

        @task
        def combine_features(numerical, categorical, derived):
            """Combine all features."""
            total = numerical["count"] + categorical["count"] + derived["count"]
            all_features = numerical["features"] + categorical["features"] + derived["features"]
            print(f"Total features: {total}")
            return {"total_features": total, "feature_names": all_features}

        num = create_numerical_features()
        cat = create_categorical_features()
        der = create_derived_features()
        return combine_features(num, cat, der)

    # ==================== MODEL TRAINING ====================
    @task
    def train_model(features: dict, pipeline_info: dict):
        """Train model with MLflow tracking."""
        import os

        print(f"Training model with {features['total_features']} features...")

        try:
            import mlflow
            import numpy as np
            from sklearn.ensemble import GradientBoostingClassifier
            from sklearn.model_selection import train_test_split, cross_val_score

            tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment("complete_ml_pipeline")

            # Generate synthetic data
            np.random.seed(42)
            n_samples = 5000
            n_features = features["total_features"]
            X = np.random.randn(n_samples, n_features)
            y = (X[:, 0] + X[:, 1] * 0.5 > 0).astype(int)

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

            params = {
                "n_estimators": 100,
                "max_depth": 4,
                "learning_rate": 0.1,
                "random_state": 42,
            }

            with mlflow.start_run(run_name=f"pipeline_{pipeline_info['pipeline_run_id']}"):
                mlflow.log_params(params)
                mlflow.log_param("n_features", n_features)
                mlflow.log_param("pipeline_run_id", pipeline_info["pipeline_run_id"])

                model = GradientBoostingClassifier(**params)

                # Cross-validation
                cv_scores = cross_val_score(model, X_train, y_train, cv=5)
                mlflow.log_metric("cv_mean", cv_scores.mean())
                mlflow.log_metric("cv_std", cv_scores.std())

                # Final training
                model.fit(X_train, y_train)

                # Test metrics
                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

                y_pred = model.predict(X_test)
                metrics = {
                    "accuracy": accuracy_score(y_test, y_pred),
                    "precision": precision_score(y_test, y_pred),
                    "recall": recall_score(y_test, y_pred),
                    "f1": f1_score(y_test, y_pred),
                }
                mlflow.log_metrics(metrics)

                # Log model
                mlflow.sklearn.log_model(model, "model")

                run_id = mlflow.active_run().info.run_id
                print(f"Training complete. Run ID: {run_id}")
                print(f"Metrics: {metrics}")

                return {
                    "run_id": run_id,
                    "metrics": metrics,
                    "model_uri": f"runs:/{run_id}/model",
                }

        except ImportError as e:
            print(f"MLflow/sklearn not available: {e}")
            # Return mock results for demo
            return {
                "run_id": "mock_run",
                "metrics": {"accuracy": 0.85, "precision": 0.83, "recall": 0.87, "f1": 0.85},
                "model_uri": "mock://model",
            }

    # ==================== MODEL EVALUATION ====================
    @task.branch
    def evaluate_model(training_result: dict):
        """Evaluate model and decide deployment path."""
        metrics = training_result["metrics"]
        accuracy = metrics["accuracy"]
        f1 = metrics["f1"]

        print(f"Evaluating model: accuracy={accuracy:.4f}, f1={f1:.4f}")

        # Decision criteria
        if accuracy >= 0.8 and f1 >= 0.75:
            print("Model approved for production!")
            return "deploy_to_production"
        elif accuracy >= 0.7:
            print("Model approved for staging only")
            return "deploy_to_staging"
        else:
            print("Model rejected - needs improvement")
            return "model_rejected"

    @task
    def deploy_to_production(training_result: dict):
        """Deploy model to production."""
        print(f"Deploying model {training_result['run_id']} to PRODUCTION")
        # In production: update model endpoint, notify stakeholders
        return {"environment": "production", "status": "deployed"}

    @task
    def deploy_to_staging(training_result: dict):
        """Deploy model to staging."""
        print(f"Deploying model {training_result['run_id']} to STAGING")
        return {"environment": "staging", "status": "deployed"}

    @task
    def model_rejected(training_result: dict):
        """Handle rejected model."""
        print(f"Model {training_result['run_id']} REJECTED - metrics below threshold")
        return {"status": "rejected", "action": "retrain_with_more_data"}

    @task(trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS)
    def pipeline_complete(deployment_result=None):
        """Complete pipeline and log summary."""
        print("=" * 50)
        print("PIPELINE COMPLETE")
        if deployment_result:
            print(f"Deployment: {deployment_result}")
        print("=" * 50)
        return {"status": "complete"}

    # ==================== PIPELINE ORCHESTRATION ====================
    pipeline_info = start_pipeline()
    data = data_ingestion()
    validation = data_validation(data)
    features = feature_engineering(validation)
    training = train_model(features, pipeline_info)
    decision = evaluate_model(training)

    prod = deploy_to_production(training)
    staging = deploy_to_staging(training)
    rejected = model_rejected(training)

    decision >> [prod, staging, rejected]
    [prod, staging, rejected] >> pipeline_complete()


complete_ml_pipeline()
