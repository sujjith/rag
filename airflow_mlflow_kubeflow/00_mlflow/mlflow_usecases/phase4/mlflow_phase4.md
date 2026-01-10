# MLflow Phase 4: Model Serving

 This folder demonstrates how to prepare models for production and test the MLflow Model Serving endpoint.

 ## Files Overview

 ### 1. `py/01_prepare_model.py`
 **Goal:** Create and register a production-ready model for serving.
 - Trains a robust model (RandomForest).
 - Logs it with a **signature** (required for serving to validate inputs).
 - Registers it to the Model Registry.
 - **Promotes to Production:** Explicitly transitions the model to the `Production` stage.
 - *Note:* This script is a prerequisite for serving. You must have a model in the `Production` stage to serve it using the standard command.

 ### 2. `py/02_test_serving.py`
 **Goal:** content-test the REST API of a served model.
 - Assumption: You have started the model server (e.g., via `mlflow models serve` or a script).
 - **API Endpoint:** Sends HTTP POST requests to `http://localhost:5001/invocations`.
 - **Input Formats:** Demonstrates 3 supported JSON payloads:
   1. `inputs`: Simple array of arrays (e.g., `[[5.1, 3.5, ...]]`).
   2. `dataframe_split`: Object with `columns` and `data` fields.
   3. `dataframe_records`: List of dictionaries (e.g., `[{"sepal length": 5.1, ...}]`).
 - **Performance:** Tests batch inference speed.
 - **Error Handling:** Shows what happens when providing invalid data shapes or types.

 ## How to Run
 1. **Prepare the model:**
    ```bash
    python py/01_prepare_model.py
    ```

 2. **Start the Model Server:**
    (Open a separate terminal)
    ```bash
    # Serve the model currently in 'Production' stage on port 5001
    mlflow models serve -m "models:/iris-serving-model/Production" -p 5001 --no-conda
    ```

 3. **Run the Tests:**
    ```bash
    python py/02_test_serving.py
    ```
