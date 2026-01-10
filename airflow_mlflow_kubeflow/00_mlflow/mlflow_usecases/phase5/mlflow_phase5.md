# MLflow Phase 5: Autologging and Custom Metrics

 This folder demonstrates two approaches to logging: automatic logging for convenience and manual logging for total control.

 ## Files Overview

 ### 1. `py/01_autolog.py`
 **Goal:** Enable automatic logging to reduce boilerplate code.
 - **Autologging:** Uses `mlflow.sklearn.autolog()` to automatically capture:
   - Parameters (e.g., `n_estimators`, `max_depth`).
   - Metrics (e.g., accuracy, loss).
   - Model artifacts.
   - Signatures and input examples.
 - **Hyperparameter Tuning:** When used with `GridSearchCV`, it creates a **parent run** for the search and **child runs** for each parameter combination automatically.
 - **Configuration:** Shows how to customize what gets logged (e.g., `log_input_examples=True`).

 ### 2. `py/02_custom_metrics.py`
 **Goal:** Log advanced/custom metrics and visualizations manually.
 - **Advanced Metrics:** Logs specific metrics like weighted F1-score, per-class precision/recall, and AUC-ROC.
 - **Visualizations:** Generates and logs plots as artifacts:
   - Confusion Matrix heatmap.
   - Feature Importance bar chart.
   - Prediction vs Actual distribution plots.
 - **Reports:** Saves JSON summaries of the evaluation.
 - This approach gives you full flexibility to evaluate models exactly how you need.

 ## How to Run
 ```bash
 python py/01_autolog.py
 python py/02_custom_metrics.py
 ```
