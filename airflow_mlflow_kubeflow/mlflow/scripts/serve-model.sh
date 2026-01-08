#!/bin/bash
set -e

MODEL_NAME=${1:-""}
MODEL_VERSION=${2:-""}
PORT=${3:-5001}

if [ -z "$MODEL_NAME" ]; then
    echo "Usage: ./serve-model.sh <model-name> [version] [port]"
    echo ""
    echo "Examples:"
    echo "  ./serve-model.sh iris-classifier 1"
    echo "  ./serve-model.sh iris-classifier Production"
    echo "  ./serve-model.sh iris-classifier 1 5002"
    exit 1
fi

# Build model URI
if [ -z "$MODEL_VERSION" ]; then
    MODEL_URI="models:/${MODEL_NAME}/Production"
else
    # Check if version is a number or stage name
    if [[ "$MODEL_VERSION" =~ ^[0-9]+$ ]]; then
        MODEL_URI="models:/${MODEL_NAME}/${MODEL_VERSION}"
    else
        MODEL_URI="models:/${MODEL_NAME}/${MODEL_VERSION}"
    fi
fi

echo "=========================================="
echo "  Starting MLflow Model Server"
echo "=========================================="
echo ""
echo "  Model:  ${MODEL_NAME}"
echo "  URI:    ${MODEL_URI}"
echo "  Port:   ${PORT}"
echo ""

# Check if container already exists
if docker ps -a --format '{{.Names}}' | grep -q "^mlflow-model-server$"; then
    echo "Stopping existing model server..."
    docker stop mlflow-model-server 2>/dev/null || true
    docker rm mlflow-model-server 2>/dev/null || true
fi

# Run model server
docker run -d \
    --name mlflow-model-server \
    --network mlflow_mlflow-network \
    -p ${PORT}:${PORT} \
    -e MLFLOW_TRACKING_URI=http://mlflow-server:5000 \
    -v mlflow_mlartifacts:/mlartifacts \
    ghcr.io/mlflow/mlflow:v2.9.2 \
    mlflow models serve \
    --model-uri "${MODEL_URI}" \
    --host 0.0.0.0 \
    --port ${PORT} \
    --no-conda

echo ""
echo "Waiting for model server to start..."
sleep 5

# Check if server is running
if docker ps --format '{{.Names}}' | grep -q "^mlflow-model-server$"; then
    echo ""
    echo "=========================================="
    echo "  Model Server Started Successfully"
    echo "=========================================="
    echo ""
    echo "  Endpoint: http://localhost:${PORT}/invocations"
    echo ""
    echo "  Test with:"
    echo "  curl -X POST http://localhost:${PORT}/invocations \\"
    echo "    -H 'Content-Type: application/json' \\"
    echo "    -d '{\"inputs\": [[5.1, 3.5, 1.4, 0.2]]}'"
    echo ""
    echo "  To stop: docker stop mlflow-model-server"
    echo "=========================================="
else
    echo "ERROR: Model server failed to start"
    echo "Check logs: docker logs mlflow-model-server"
    exit 1
fi
