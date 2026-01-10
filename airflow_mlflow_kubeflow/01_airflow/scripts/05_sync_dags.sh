#!/bin/bash
set -e

echo "=========================================="
echo "  Sync DAGs to Airflow"
echo "=========================================="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DAGS_DIR="${SCRIPT_DIR}/../dags"
NAMESPACE=${NAMESPACE:-airflow}

# Find the scheduler pod
SCHEDULER_POD=$(kubectl get pods -n ${NAMESPACE} -l component=scheduler -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)

if [ -z "$SCHEDULER_POD" ]; then
    echo "ERROR: Could not find scheduler pod"
    exit 1
fi

echo ""
echo "Scheduler pod: ${SCHEDULER_POD}"
echo "DAGs directory: ${DAGS_DIR}"
echo ""

# Copy DAGs to scheduler
echo "Copying DAGs..."
for phase_dir in ${DAGS_DIR}/phase*; do
    if [ -d "$phase_dir" ]; then
        phase_name=$(basename "$phase_dir")
        echo "  Syncing ${phase_name}..."

        for dag_file in ${phase_dir}/*.py; do
            if [ -f "$dag_file" ]; then
                dag_name=$(basename "$dag_file")
                kubectl cp "$dag_file" "${NAMESPACE}/${SCHEDULER_POD}:/opt/airflow/dags/${dag_name}"
                echo "    - ${dag_name}"
            fi
        done
    fi
done

# Also copy to workers
echo ""
echo "Syncing to workers..."
for worker_pod in $(kubectl get pods -n ${NAMESPACE} -l component=worker -o jsonpath='{.items[*].metadata.name}'); do
    echo "  Worker: ${worker_pod}"
    for phase_dir in ${DAGS_DIR}/phase*; do
        if [ -d "$phase_dir" ]; then
            for dag_file in ${phase_dir}/*.py; do
                if [ -f "$dag_file" ]; then
                    dag_name=$(basename "$dag_file")
                    kubectl cp "$dag_file" "${NAMESPACE}/${worker_pod}:/opt/airflow/dags/${dag_name}" 2>/dev/null || true
                fi
            done
        fi
    done
done

echo ""
echo "=========================================="
echo "  DAGs Synced"
echo "=========================================="
echo ""
echo "Wait 30-60 seconds for scheduler to parse DAGs"
echo ""
echo "List DAGs:"
echo "  kubectl exec -n ${NAMESPACE} ${SCHEDULER_POD} -- airflow dags list"
echo "=========================================="
