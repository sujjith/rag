#Phase 1.5: Hyperparameter Search - WITH DETAILED EXPLANATIONS

import mlflow
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import os

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment("phase1-hyperparameter-search-explained")

print("=" * 80)
print("STEP 1: LOAD THE IRIS DATASET")
print("=" * 80)

# Load data
X, y = load_iris(return_X_y=True)
print(f"✓ Loaded Iris dataset")
print(f"  - Total samples: {len(X)}")
print(f"  - Features per sample: {X.shape[1]} (sepal length, sepal width, petal length, petal width)")
print(f"  - Classes: 3 (setosa=0, versicolor=1, virginica=2)")
print(f"\nFirst 3 samples:")
print(f"  Sample 1: {X[0]} → Class {y[0]}")
print(f"  Sample 2: {X[1]} → Class {y[1]}")
print(f"  Sample 3: {X[2]} → Class {y[2]}")

print("\n" + "=" * 80)
print("STEP 2: SPLIT DATA INTO TRAINING AND TEST SETS")
print("=" * 80)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"✓ Split complete!")
print(f"  - Training set: {len(X_train)} samples (80%) - MODEL WILL LEARN FROM THESE")
print(f"  - Test set: {len(X_test)} samples (20%) - MODEL HAS NEVER SEEN THESE")
print(f"\nWhy split?")
print(f"  → Training data: Teach the model patterns")
print(f"  → Test data: Check if model REALLY learned (not just memorized)")

print("\n" + "=" * 80)
print("STEP 3: TRY DIFFERENT HYPERPARAMETERS")
print("=" * 80)

# Just test ONE configuration for clarity
params = {"n_estimators": 50, "max_depth": 5, "min_samples_split": 2}

print(f"\nHyperparameters to try:")
print(f"  - n_estimators: {params['n_estimators']} (number of decision trees)")
print(f"  - max_depth: {params['max_depth']} (how deep each tree can grow)")
print(f"  - min_samples_split: {params['min_samples_split']} (min samples to split a node)")

with mlflow.start_run(run_name="explained-run"):
    print("\n" + "-" * 80)
    print("STEP 4: CREATE AND TRAIN THE MODEL")
    print("-" * 80)
    
    # Create model
    model = RandomForestClassifier(**params, random_state=42)
    print(f"✓ Created RandomForest model with parameters: {params}")
    
    # Train model
    print(f"\n🔄 TRAINING STARTED...")
    print(f"   Model is looking at {len(X_train)} flowers and learning patterns...")
    model.fit(X_train, y_train)
    print(f"✓ TRAINING COMPLETE!")
    print(f"   Model has learned patterns from the training data")
    
    print("\n" + "-" * 80)
    print("STEP 5: TEST THE MODEL - DID IT REALLY LEARN?")
    print("-" * 80)
    
    # Predict on training data
    print(f"\n📊 Testing on TRAINING data (data model has seen):")
    y_pred_train = model.predict(X_train)
    print(f"   Model made {len(y_pred_train)} predictions")
    print(f"   First 5 predictions: {y_pred_train[:5]}")
    print(f"   First 5 actual:      {y_train[:5]}")
    
    train_acc = accuracy_score(y_train, y_pred_train)
    correct_train = int(train_acc * len(y_train))
    print(f"\n   ✓ Training Accuracy: {train_acc:.4f} ({correct_train}/{len(y_train)} correct)")
    
    # Predict on test data - THE REAL TEST!
    print(f"\n📊 Testing on TEST data (NEW data model has NEVER seen):")
    y_pred_test = model.predict(X_test)
    print(f"   Model made {len(y_pred_test)} predictions on flowers it never saw before")
    print(f"   First 5 predictions: {y_pred_test[:5]}")
    print(f"   First 5 actual:      {y_test[:5]}")
    
    test_acc = accuracy_score(y_test, y_pred_test)
    correct_test = int(test_acc * len(y_test))
    print(f"\n   ✓ Test Accuracy: {test_acc:.4f} ({correct_test}/{len(y_test)} correct)")
    
    print("\n" + "-" * 80)
    print("STEP 6: WHAT DOES THIS MEAN?")
    print("-" * 80)
    
    if test_acc > 0.9:
        print(f"\n🎉 EXCELLENT! Test accuracy is {test_acc:.1%}")
        print(f"   → The model LEARNED real patterns!")
        print(f"   → It correctly classified {correct_test} out of {len(y_test)} flowers it never saw")
        print(f"   → This is NOT random guessing (would be ~33%)")
        print(f"   → This is NOT memorization (test data is NEW)")
    elif test_acc > 0.7:
        print(f"\n👍 GOOD! Test accuracy is {test_acc:.1%}")
        print(f"   → The model learned some patterns")
    else:
        print(f"\n⚠️  LOW! Test accuracy is {test_acc:.1%}")
        print(f"   → The model didn't learn well")
    
    # Show some specific predictions
    print(f"\n📋 Let's look at some specific predictions:")
    for i in range(min(5, len(y_test))):
        predicted = y_pred_test[i]
        actual = y_test[i]
        class_names = ['setosa', 'versicolor', 'virginica']
        status = "✓ CORRECT" if predicted == actual else "✗ WRONG"
        print(f"   Flower {i+1}: Predicted={class_names[predicted]}, Actual={class_names[actual]} {status}")
    
    # Log to MLflow
    print("\n" + "-" * 80)
    print("STEP 7: LOG RESULTS TO MLFLOW")
    print("-" * 80)
    
    mlflow.log_param("n_estimators", params['n_estimators'])
    mlflow.log_param("max_depth", params['max_depth'])
    mlflow.log_param("min_samples_split", params['min_samples_split'])
    print(f"✓ Logged parameters to MLflow")
    
    mlflow.log_metric("train_accuracy", train_acc)
    mlflow.log_metric("test_accuracy", test_acc)
    print(f"✓ Logged metrics to MLflow")
    
    mlflow.set_tag("model_type", "RandomForest")
    print(f"✓ Logged tags to MLflow")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"✓ Loaded {len(X)} iris flowers")
print(f"✓ Split into {len(X_train)} training + {len(X_test)} test samples")
print(f"✓ Trained RandomForest model on training data")
print(f"✓ Tested on NEW data the model never saw")
print(f"✓ Achieved {test_acc:.1%} accuracy on test data")
print(f"\n🎯 KEY INSIGHT:")
print(f"   The model got {correct_test}/{len(y_test)} correct on flowers it NEVER saw during training.")
print(f"   This proves it learned the PATTERNS, not just memorized the data!")
print(f"\n🌐 View in MLflow UI: {TRACKING_URI}")
print("=" * 80)
