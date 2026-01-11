
from kfp import dsl
from kfp import compiler
from kfp.client import Client
import kfp

@dsl.component(packages_to_install=['pandas', 'boto3', 's3fs', 'pyarrow'])
def load_data(
    bucket_name: str,
    object_name: str,
    s3_endpoint: str,
    s3_access_key: str,
    s3_secret_key: str,
) -> dsl.Dataset:
    import pandas as pd
    import boto3
    from botocore.client import Config
    import os
    from io import StringIO
    
    print(f"Loading data from {bucket_name}/{object_name}")
    
    s3 = boto3.client('s3',
                      endpoint_url=s3_endpoint,
                      aws_access_key_id=s3_access_key,
                      aws_secret_access_key=s3_secret_key,
                      config=Config(signature_version='s3v4'),
                      region_name='us-east-1')

    # Read CSV
    obj = s3.get_object(Bucket=bucket_name, Key=object_name)
    df = pd.read_csv(obj['Body'])
    
    print(f"Loaded {len(df)} rows")
    
    # Save to output artifact
    # dsl.Dataset is a path
    # KFP passes the path as an argument to the function (implicitly handled by decorator wrapper if mapped correctly,
    # but for output artifact we usually return the content or save to the path provided)
    # Actually KFP v2 components return outputs via return annotation or Output[Dataset]
    # Simple python components with return annotation -> serializes return value.
    # But dsl.Dataset is a file path.
    # Let's use Output[Dataset] style for clarity or return the dataframe serialized?
    # Returning dsl.Dataset directly: KFP expects we write to the path?
    # No, '-> dsl.Dataset' means we return a value that KFP writes?
    # Let's use the Output[Dataset] parameter style which is explicit.
    return df.to_csv(index=False) 

# Re-defining load_data with Output[Dataset] is safer
@dsl.component(packages_to_install=['pandas', 'boto3', 's3fs', 'pyarrow'])
def load_data_component(
    bucket_name: str,
    object_name: str,
    s3_endpoint: str,
    s3_access_key: str,
    s3_secret_key: str,
    dataset: dsl.Output[dsl.Dataset]
):
    import pandas as pd
    import boto3
    from botocore.client import Config
    
    print(f"Loading from s3://{bucket_name}/{object_name}")
    s3 = boto3.client('s3',
                      endpoint_url=s3_endpoint,
                      aws_access_key_id=s3_access_key,
                      aws_secret_access_key=s3_secret_key,
                      config=Config(signature_version='s3v4'),
                      region_name='us-east-1')

    obj = s3.get_object(Bucket=bucket_name, Key=object_name)
    df = pd.read_csv(obj['Body'])
    
    # Write to the output path provided by KFP
    df.to_csv(dataset.path, index=False)
    print(f"Saved dataset to {dataset.path}")

@dsl.component(packages_to_install=['pandas', 'scikit-learn', 'joblib'])
def train_model_component(
    dataset: dsl.Input[dsl.Dataset],
    model: dsl.Output[dsl.Model],
    metrics: dsl.Output[dsl.Metrics]
):
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, precision_score, recall_score
    import joblib
    
    # Load data
    df = pd.read_csv(dataset.path)
    
    # Preprocessing (simplified)
    # Drop timestamp columns if they exist
    cols_to_drop = ['customer_id', 'event_timestamp', 'created_timestamp']
    for col in cols_to_drop:
        if col in df.columns:
            df = df.drop(columns=[col])
            
    # Simple encoding
    df = pd.get_dummies(df)
    
    X = df.drop('churn', axis=1)
    y = df['churn']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    
    # Save model
    joblib.dump(clf, model.path) # KFP Model artifact is usually a file or dir
    # If model.path assumes a file name, joblib.dump works. If dir, we need to join.
    # Usually dsl.Model path ends with filename or we create it.
    # Let's ensure it's saved.
    
    # Log metrics
    metrics.log_metric("accuracy", acc)
    metrics.log_metric("precision", prec)
    metrics.log_metric("recall", rec)
    
    print(f"Model trained. Accuracy: {acc}")

@dsl.component(packages_to_install=['mlflow', 'boto3', 'joblib', 'scikit-learn', 'pandas'])
def register_model_component(
    model: dsl.Input[dsl.Model],
    mlflow_tracking_uri: str,
    experiment_name: str,
    s3_endpoint: str,
    s3_access_key: str,
    s3_secret_key: str,
):
    import mlflow
    import joblib
    import os
    
    # Set environment variables for MLflow S3 access
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = s3_endpoint
    os.environ["AWS_ACCESS_KEY_ID"] = s3_access_key
    os.environ["AWS_SECRET_ACCESS_key"] = s3_secret_key
    os.environ["AWS_DEFAULT_REGION"] = "us-east-1"
    
    print(f"Connecting to MLflow at {mlflow_tracking_uri}")
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(experiment_name)
    
    # Load model from artifact
    loaded_model = joblib.load(model.path)
    
    with mlflow.start_run():
        mlflow.sklearn.log_model(loaded_model, "model")
        print("Model logged to MLflow")

@dsl.pipeline(
    name='Churn Prediction Pipeline',
    description='End-to-end churn prediction pipeline'
)
def churn_pipeline(
    bucket_name: str = 'ml-data',
    object_name: str = 'customers.csv',
    s3_endpoint: str = 'http://minio.minio.svc.cluster.local:9000',
    s3_access_key: str = 'minioadmin',
    s3_secret_key: str = 'minioadmin123',
    mlflow_tracking_uri: str = 'http://mlflow.mlflow.svc.cluster.local:5000',
    experiment_name: str = 'churn_prediction_v1'
):
    load_task = load_data_component(
        bucket_name=bucket_name,
        object_name=object_name,
        s3_endpoint=s3_endpoint,
        s3_access_key=s3_access_key,
        s3_secret_key=s3_secret_key
    )
    
    train_task = train_model_component(
        dataset=load_task.outputs['dataset']
    )
    
    register_task = register_model_component(
        model=train_task.outputs['model'],
        mlflow_tracking_uri=mlflow_tracking_uri,
        experiment_name=experiment_name,
        s3_endpoint=s3_endpoint,
        s3_access_key=s3_access_key,
        s3_secret_key=s3_secret_key
    )

if __name__ == '__main__':
    # compile
    compiler.Compiler().compile(churn_pipeline, 'churn_pipeline.yaml')
    print("Pipeline compiled to churn_pipeline.yaml")
    
    # submit (optional, if we want to run it from here)
    try:
        # Connecting to localhost:3000 (UI port-forward) which might not expose API directly at root
        # Usually client connects to API. 
        # Attempt connection
        client = Client(host='http://localhost:3000')
        run = client.create_run_from_pipeline_func(
            churn_pipeline,
            arguments={},
            experiment_name='churn_experiment'
        )
        print(f"Run submitted: {run.run_id}")
        print(f"Run link: http://localhost:3000/#/runs/details/{run.run_id}")
    except Exception as e:
        print(f"Could not submit run automatically: {e}")
        print("You can upload churn_pipeline.yaml to the KFP UI manually.")
