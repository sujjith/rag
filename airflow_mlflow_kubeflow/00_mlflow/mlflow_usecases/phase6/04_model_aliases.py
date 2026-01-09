"""
Phase 6.4: Model Aliases (Enterprise Pattern - MLflow 2.3+)

This script demonstrates:
- Creating and managing model aliases
- Semantic versioning (latest, stable, canary, etc.)
- Alias-based model loading
- Safe deployment patterns with aliases
- Migration from stages to aliases

Note: Model aliases were introduced in MLflow 2.3+ as a more flexible
alternative to model stages (None/Staging/Production/Archived).

Run: python 04_model_aliases.py
"""
import mlflow
from mlflow.tracking import MlflowClient
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import os
import time

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment("phase6-model-aliases")

client = MlflowClient()
MODEL_NAME = "iris-classifier-aliased"

# Load data
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Clean up existing model
try:
    client.delete_registered_model(MODEL_NAME)
except:
    pass

print("=" * 70)
print("Model Aliases - Enterprise Versioning")
print("=" * 70)

print("""
  Model Aliases vs Stages:

  Stages (Legacy)          Aliases (Modern)
  ----------------         ----------------
  - None                   - @latest
  - Staging                - @candidate
  - Production             - @champion
  - Archived               - @v1.0, @v2.0 (semantic)
                           - @stable, @experimental

  Benefits of Aliases:
  - Multiple aliases per version
  - Custom naming (semantic versioning)
  - No stage transitions needed
  - Better for GitOps workflows
""")


# [1] Create Multiple Model Versions
print("\n[1] Creating Multiple Model Versions...")
print("-" * 50)

configs = [
    {"n_estimators": 50, "max_depth": 3, "version_tag": "v1.0"},
    {"n_estimators": 100, "max_depth": 5, "version_tag": "v1.1"},
    {"n_estimators": 150, "max_depth": 7, "version_tag": "v2.0"},
]

versions = {}

for config in configs:
    with mlflow.start_run(run_name=f"model-{config['version_tag']}"):
        model = RandomForestClassifier(
            n_estimators=config["n_estimators"],
            max_depth=config["max_depth"],
            random_state=42
        )
        model.fit(X_train, y_train)

        accuracy = accuracy_score(y_test, model.predict(X_test))
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_param("n_estimators", config["n_estimators"])
        mlflow.log_param("max_depth", config["max_depth"])
        mlflow.set_tag("version_tag", config["version_tag"])

        result = mlflow.sklearn.log_model(
            model, "model",
            registered_model_name=MODEL_NAME
        )

        print(f"  {config['version_tag']}: accuracy={accuracy:.4f}")

    time.sleep(1)

# Get all versions
all_versions = client.search_model_versions(f"name='{MODEL_NAME}'")
for mv in all_versions:
    versions[mv.version] = mv
    print(f"    Registered: version {mv.version}")


# [2] Set Model Aliases
print("\n[2] Setting Model Aliases...")
print("-" * 50)

# Check MLflow version for alias support
try:
    # MLflow 2.3+ supports aliases
    # Set semantic version aliases
    client.set_registered_model_alias(MODEL_NAME, "v1.0", "1")
    client.set_registered_model_alias(MODEL_NAME, "v1.1", "2")
    client.set_registered_model_alias(MODEL_NAME, "v2.0", "3")

    # Set deployment aliases
    client.set_registered_model_alias(MODEL_NAME, "stable", "2")      # v1.1 is stable
    client.set_registered_model_alias(MODEL_NAME, "champion", "2")    # Same as stable
    client.set_registered_model_alias(MODEL_NAME, "candidate", "3")   # v2.0 is candidate
    client.set_registered_model_alias(MODEL_NAME, "latest", "3")      # Latest version

    print("  Aliases set:")
    print("    @v1.0 -> version 1 (initial release)")
    print("    @v1.1 -> version 2 (bug fix)")
    print("    @v2.0 -> version 3 (major update)")
    print("    @stable -> version 2 (production ready)")
    print("    @champion -> version 2 (current production)")
    print("    @candidate -> version 3 (testing)")
    print("    @latest -> version 3 (newest)")

    ALIASES_SUPPORTED = True

except AttributeError:
    print("  WARNING: Model aliases require MLflow 2.3+")
    print("  Your MLflow version doesn't support aliases.")
    print("  Demonstrating alias patterns conceptually...")
    ALIASES_SUPPORTED = False


# [3] Load Models by Alias
print("\n[3] Loading Models by Alias...")
print("-" * 50)

if ALIASES_SUPPORTED:
    alias_examples = ["champion", "candidate", "stable", "v1.0"]

    for alias in alias_examples:
        try:
            model_uri = f"models:/{MODEL_NAME}@{alias}"
            loaded_model = mlflow.pyfunc.load_model(model_uri)

            # Get the actual version for this alias
            model_version = client.get_model_version_by_alias(MODEL_NAME, alias)

            predictions = loaded_model.predict(X_test[:3])
            print(f"  @{alias} (version {model_version.version}): predictions={list(predictions)}")

        except Exception as e:
            print(f"  @{alias}: Error - {str(e)[:50]}")
else:
    print("  Using version numbers instead (alias fallback):")
    for version in ["1", "2", "3"]:
        model_uri = f"models:/{MODEL_NAME}/{version}"
        loaded_model = mlflow.pyfunc.load_model(model_uri)
        predictions = loaded_model.predict(X_test[:3])
        print(f"  version {version}: predictions={list(predictions)}")


# [4] Alias-Based Deployment Workflow
print("\n[4] Alias-Based Deployment Workflow...")
print("-" * 50)

print("""
  Recommended Alias Strategy:

  Development:
    @dev      -> Latest development version
    @test     -> Version under testing

  Staging:
    @candidate -> Promotion candidate
    @canary    -> Small traffic slice

  Production:
    @champion  -> Current production model
    @stable    -> Known stable version
    @fallback  -> Emergency rollback target

  Versioning:
    @v1.0, @v1.1, @v2.0 -> Semantic versions
    @latest    -> Most recent version
""")


# [5] Simulate Promotion Workflow
print("\n[5] Simulating Promotion Workflow...")
print("-" * 50)

if ALIASES_SUPPORTED:
    print("  Step 1: New model trained -> assign @candidate")
    print("  Step 2: Validation passes -> assign @canary (10% traffic)")
    print("  Step 3: Canary success -> assign @champion (full traffic)")
    print("  Step 4: Old champion -> assign @fallback")

    # Demonstrate promotion
    print("\n  Promoting candidate to champion...")

    # Get current champion version
    try:
        old_champion = client.get_model_version_by_alias(MODEL_NAME, "champion")
        old_champion_version = old_champion.version
    except:
        old_champion_version = "2"

    # Set old champion as fallback
    client.set_registered_model_alias(MODEL_NAME, "fallback", old_champion_version)
    print(f"    Old champion (v{old_champion_version}) -> @fallback")

    # Promote candidate to champion
    try:
        candidate = client.get_model_version_by_alias(MODEL_NAME, "candidate")
        candidate_version = candidate.version
    except:
        candidate_version = "3"

    client.set_registered_model_alias(MODEL_NAME, "champion", candidate_version)
    print(f"    Candidate (v{candidate_version}) -> @champion")

    print("\n  Current alias state:")
    print(f"    @champion -> version {candidate_version} (new production)")
    print(f"    @fallback -> version {old_champion_version} (rollback target)")
    print(f"    @stable -> version 2 (unchanged)")
else:
    print("  (Alias operations skipped - MLflow 2.3+ required)")


# [6] Emergency Rollback with Aliases
print("\n[6] Emergency Rollback Pattern...")
print("-" * 50)

if ALIASES_SUPPORTED:
    print("""
  Rollback using aliases (instant, no stage transitions):

  ```python
  # Get fallback version
  fallback = client.get_model_version_by_alias(MODEL_NAME, "fallback")

  # Instant rollback - just move the alias
  client.set_registered_model_alias(MODEL_NAME, "champion", fallback.version)

  # Production immediately uses the fallback model
  model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@champion")
  ```

  Advantages over stages:
  - No waiting for stage transition
  - Multiple rollback targets available
  - Audit trail preserved (alias history)
  - No need to archive current version
""")
else:
    print("  (Rollback demonstration skipped - MLflow 2.3+ required)")


# [7] List All Aliases
print("\n[7] Listing All Aliases...")
print("-" * 50)

if ALIASES_SUPPORTED:
    try:
        registered_model = client.get_registered_model(MODEL_NAME)

        # Get aliases from model versions
        print(f"  Model: {MODEL_NAME}")
        print("  Aliases:")

        all_versions = client.search_model_versions(f"name='{MODEL_NAME}'")
        for mv in all_versions:
            aliases = mv.aliases if hasattr(mv, 'aliases') else []
            alias_str = ", ".join([f"@{a}" for a in aliases]) if aliases else "(no aliases)"
            print(f"    Version {mv.version}: {alias_str}")

    except Exception as e:
        print(f"  Error listing aliases: {e}")
else:
    print("  (Alias listing skipped - MLflow 2.3+ required)")


# [8] Clean Up Aliases
print("\n[8] Managing Aliases...")
print("-" * 50)

if ALIASES_SUPPORTED:
    print("""
  Alias Management Operations:

  # Set alias
  client.set_registered_model_alias(name, alias, version)

  # Get version by alias
  mv = client.get_model_version_by_alias(name, alias)

  # Delete alias
  client.delete_registered_model_alias(name, alias)

  # Load model by alias
  model = mlflow.pyfunc.load_model(f"models:/{name}@{alias}")
""")
else:
    print("  (Management examples shown for reference)")


print("\n" + "=" * 70)
print("Model Aliases Complete!")
print("=" * 70)
print(f"""
  Key Takeaways:

  1. Aliases provide flexible, semantic versioning
  2. Multiple aliases can point to the same version
  3. Instant promotions/rollbacks (no stage transitions)
  4. Better suited for GitOps and CI/CD workflows
  5. Coexists with legacy stages during migration

  Migration Path:
  - Keep using stages for existing workflows
  - Add aliases for new deployment patterns
  - Gradually migrate to alias-only workflow

  Production Usage:
  ```python
  # Always load by alias in production code
  champion = mlflow.pyfunc.load_model("models:/my-model@champion")

  # Never hardcode version numbers
  # BAD: model = load_model("models:/my-model/3")
  ```

  View at: {TRACKING_URI}
""")
print("=" * 70)
