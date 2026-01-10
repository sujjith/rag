# MLflow Phase 1: Getting Started

 This folder contains introductory scripts to help you get started with MLflow tracking.

 ## Files Overview

 ### 1. `01_hello_mlflow.py`
 **Goal:** specific basic MLflow connectivity and logging.
 - Connects to the MLflow tracking server.
 - Creates an experiment named `phase1-hello-mlflow`.
 - Logs single values:
   - **Parameters** (`log_param`): Inputs like `learning_rate`, `epochs`.
   - **Metrics** (`log_metric`): Outputs like `accuracy`, `loss`.
   - **Tags** (`set_tag`): Metadata like `author`, `environment`.

 ### 2. `02_training_simulation.py`
 **Goal:** Demonstrate logging metrics over time (epochs).
 - Simulates a model training loop where accuracy increases and loss decreases.
 - Uses `mlflow.log_metric(..., step=epoch)` to log values at each step.
 - This allows you to view **charts and curves** in the MLflow UI (e.g., Accuracy vs Epochs).

 ### 3. `03_hyperparameter_search.py`
 **Goal:** Run and compare multiple experiments.
 - Uses `sklearn` (Iris dataset and RandomForest).
 - Defines a `param_grid` with different hyperparameter combinations (number of trees, depth, etc.).
 - Iterates through the grid, creating a **new MLflow run** for each configuration.
 - Logs parameters and resulting metrics (accuracy) for each run.
 - This enables the **Compare Runs** feature in MLflow UI to find the best hyperparameters.

 ## How to Run
 ```bash
 python 01_hello_mlflow.py
 python 02_training_simulation.py
 python 03_hyperparameter_search.py
 ```
