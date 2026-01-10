# MLflow Phase 2: Logging Artifacts and Models

 This folder focuses on logging files (artifacts), models, and understanding model metadata like signatures and flavors.

 ## Files Overview

 ### 1. `py/01_artifacts.py`
 **Goal:** Log non-model files (artifacts) to tracking.
 - **Plots:** Log matplotlib images (`log_artifact`).
 - **Data:** Log CSV files.
 - **Configs:** Log JSON configurations.
 - **Directories:** Log entire folders using `log_artifacts`.
 - This is useful for saving exploratory analysis, reports, or supplementary data alongside your run.

 ### 2. `py/02_model_logging.py`
 **Goal:** Standard model logging workflow.
 - Trains a RandomForest model.
 - Logs the model using `mlflow.sklearn.log_model`.
 - Demonstrates **loading** the model back in two ways:
   1. `mlflow.sklearn.load_model()`: Returns the native sklearn object.
   2. `mlflow.pyfunc.load_model()`: Returns a generic Python function wrapper (useful for deployment).

 ### 3. `py/03_signatures.py`
 **Goal:** Define input/output schemas (Signatures) for models.
 - **Inferred Signature:** Automatically derived from input data (`infer_signature`).
 - **Manual Signature:** explicitly defining columns and types using `Schema` and `ColSpec`.
 - **Tensor Signature:** For array-based inputs (common in Deep Learning).
 - Signatures ensure your model checks input types and names at inference time, preventing errors.

 ### 4. `py/04_model_flavors.py`
 **Goal:** Explore different model types and custom logic.
 - **Sklearn Flavor:** Logs standard models like RandomForest, GradientBoosting.
 - **Pipelines:** Logs sklearn Pipelines (preprocessing + modeling) as a single artifact.
 - **Custom PyFunc:** Defines a custom class inheriting from `mlflow.pyfunc.PythonModel` to encapsulate custom logic (e.g., preprocessing inside the model, returning class names instead of indices).

 ## How to Run
 ```bash
 python py/01_artifacts.py
 python py/02_model_logging.py
 python py/03_signatures.py
 python py/04_model_flavors.py
 ```
