#!/bin/bash
# Infrastructure Verification Script

echo "=== Phase 01 Infrastructure Verification ==="

echo -e "\n1. Docker Status:"
docker info | grep -E "Server Version|Storage Driver|Cgroup Driver"

echo -e "\n2. Kubernetes Cluster:"
kubectl cluster-info
kubectl get nodes -o wide

echo -e "\n3. Namespaces:"
kubectl get namespaces | grep -E "airflow|mlflow|kubeflow|monitoring|feast|cert-manager|kserve"

echo -e "\n4. Storage Classes:"
kubectl get storageclass

echo -e "\n5. PostgreSQL Status:"
kubectl get pods -n mlflow -l app.kubernetes.io/name=postgresql

echo -e "\n6. MinIO Status:"
kubectl get pods -n mlflow -l app=minio

echo -e "\n7. Redis Status:"
kubectl get pods -n airflow -l app.kubernetes.io/name=redis

echo -e "\n8. Ingress Controller:"
kubectl get pods -n ingress-nginx

echo -e "\n9. Cert-Manager:"
kubectl get pods -n cert-manager

echo -e "\n10. ClusterIssuers:"
kubectl get clusterissuers

echo -e "\n11. Secrets:"
kubectl get secrets -n mlflow | grep -E "postgres|minio"
kubectl get secrets -n airflow | grep redis

echo -e "\n12. DNS Entries:"
cat /etc/hosts | grep ".local"

echo -e "\n13. Minikube Status:"
minikube status

echo -e "\n=== All checks completed ==="
echo -e "\n✅ Phase 01 Infrastructure Setup Complete!"
echo -e "\nNext Steps:"
echo -e "  - Phase 02: MLflow Enterprise Setup"
echo -e "  - Phase 03: Airflow Enterprise Setup"
echo -e "  - Phase 04: Kubeflow Complete Setup"
