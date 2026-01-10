#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCKER_DIR="${SCRIPT_DIR}/../docker"

echo "=========================================="
echo "  Starting MLflow Platform"
echo "=========================================="

cd "${DOCKER_DIR}"

# Build and start containers
echo "[1/3] Building Docker images..."
docker-compose build

echo "[2/3] Starting services..."
docker-compose up -d

echo "[3/3] Waiting for services to be healthy..."
sleep 5

# Wait for PostgreSQL
echo "Waiting for PostgreSQL..."
until docker exec mlflow-postgres pg_isready -U mlflow -d mlflow > /dev/null 2>&1; do
    sleep 2
done
echo "PostgreSQL is ready!"

# Wait for MLflow
echo "Waiting for MLflow server..."
until curl -s http://localhost:5000/health > /dev/null 2>&1; do
    sleep 2
done
echo "MLflow server is ready!"

echo ""
echo "=========================================="
echo "  MLflow Platform Started Successfully"
echo "=========================================="
echo ""
echo "  MLflow UI:    http://localhost:5000"
echo "  PostgreSQL:   localhost:5432"
echo ""
echo "  To stop: ./scripts/stop.sh"
echo "=========================================="
