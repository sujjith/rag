# MLflow Project Example

This is a self-contained MLflow Project for Iris classification.

## Structure

```
03_mlproject/
├── MLproject              # Project definition
├── python_env.yaml        # Python environment
├── train.py               # Training entry point
├── evaluate.py            # Evaluation entry point
├── hyperparameter_search.py  # HP search entry point
└── README.md              # This file
```

## Entry Points

### 1. main (Training)

```bash
# Default parameters
mlflow run .

# Custom parameters
mlflow run . -P n_estimators=200 -P max_depth=10 -P test_size=0.3
```

### 2. evaluate

```bash
# Evaluate a previous run
mlflow run . -e evaluate -P run_id=<run_id>
```

### 3. hyperparameter_search

```bash
# Run hyperparameter search
mlflow run . -e hyperparameter_search -P n_trials=10
```

## Running from GitHub

```bash
# Run directly from a GitHub repository
mlflow run https://github.com/your-repo/mlflow-project
```

## Environment

The project uses a Python environment defined in `python_env.yaml`:
- Python 3.11
- mlflow 2.9.2
- scikit-learn 1.3.2
- pandas 2.1.4
