# MLflow Phase 3: Model Registry

 This folder demonstrates how to use the MLflow Model Registry to manage model versioning, stages, and lifecycle.

 ## Files Overview

 ### 1. `py/01_register_model.py`
 **Goal:** Register trained models into the central registry.
 - **Registering:** Two methods shown:
   1. `mlflow.register_model(uri, name)`: Register an existing run.
   2. `mlflow.sklearn.log_model(..., registered_model_name=name)`: Register immediately during training.
 - **Descriptions:** How to add documentation/descriptions to your registered models using `MlflowClient`.

 ### 2. `py/02_model_stages.py`
 **Goal:** Manage model lifecycle stages (Staging, Production, Archived).
 - Creates multiple versions of a model.
 - Uses `client.transition_model_version_stage()` to move models between stages.
 - **Stages:**
   - **None:** Initial state.
   - **Staging:** For testing candidate models.
   - **Production:** For the live/active model.
   - **Archived:** For retired models.

 ### 3. `py/03_load_from_registry.py`
 **Goal:** Load models dynamically from the registry.
 - **By Version:** `models:/<name>/<version>` (e.g., specific immutable artifact).
 - **By Stage:** `models:/<name>/Production` (dynamic reference to whatever is currently live).
 - **Best Practice:** In production code, always load from the `Production` stage alias so you don't need to change code when the model updates.

 ### 4. `py/04_registry_workflow.py`
 **Goal:** Simulate a full MLOps workflow.
 - Trains an initial model -> Registers it -> Promotes to Production.
 - Trains a "v2" model (improved).
 - Compares v2 vs Production metrics.
 - **Conditional Promotion:** Automatically promotes v2 to Production only if it performs better than the current Production model.
 - Archives the old model automatically.

 ## How to Run
 ```bash
 python py/01_register_model.py
 python py/02_model_stages.py
 python py/03_load_from_registry.py
 python py/04_registry_workflow.py
 ```
