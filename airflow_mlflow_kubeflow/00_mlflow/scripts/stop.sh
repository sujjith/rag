#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCKER_DIR="${SCRIPT_DIR}/../docker"

echo "=========================================="
echo "  Stopping MLflow Platform"
echo "=========================================="

cd "${DOCKER_DIR}"

docker-compose down

echo ""
echo "MLflow Platform stopped."
echo ""
echo "To remove volumes (data): docker-compose down -v"
