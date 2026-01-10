#!/bin/bash

echo "=========================================="
echo "  MLflow Platform Health Check"
echo "=========================================="
echo ""

# Check PostgreSQL
echo -n "[PostgreSQL]  "
if docker exec mlflow-postgres pg_isready -U mlflow -d mlflow > /dev/null 2>&1; then
    echo "OK - accepting connections"
else
    echo "FAILED - not responding"
fi

# Check MLflow Server
echo -n "[MLflow]      "
if curl -s http://localhost:5000/health > /dev/null 2>&1; then
    echo "OK - http://localhost:5000"
else
    echo "FAILED - not responding"
fi

# Check Model Server (optional)
echo -n "[Model Srv]   "
if docker ps --format '{{.Names}}' | grep -q "^mlflow-model-server$"; then
    if curl -s http://localhost:5001/health > /dev/null 2>&1; then
        echo "OK - http://localhost:5001"
    else
        echo "RUNNING - warming up"
    fi
else
    echo "NOT RUNNING"
fi

echo ""

# Show container status
echo "Container Status:"
echo "-----------------"
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | grep -E "^NAMES|mlflow"

echo ""

# Show experiment count
echo "MLflow Statistics:"
echo "------------------"
EXPERIMENTS=$(curl -s http://localhost:5000/api/2.0/mlflow/experiments/search 2>/dev/null | python3 -c "import sys, json; data=json.load(sys.stdin); print(len(data.get('experiments', [])))" 2>/dev/null || echo "N/A")
echo "  Experiments: ${EXPERIMENTS}"

MODELS=$(curl -s http://localhost:5000/api/2.0/mlflow/registered-models/search 2>/dev/null | python3 -c "import sys, json; data=json.load(sys.stdin); print(len(data.get('registered_models', [])))" 2>/dev/null || echo "N/A")
echo "  Registered Models: ${MODELS}"

echo ""
