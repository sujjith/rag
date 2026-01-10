#!/bin/bash
# Add DNS entries for local development

INGRESS_IP="192.168.49.2"

echo "Adding DNS entries to /etc/hosts..."
sudo tee -a /etc/hosts <<EOF
${INGRESS_IP} airflow.local
${INGRESS_IP} mlflow.local
${INGRESS_IP} kubeflow.local
${INGRESS_IP} grafana.local
${INGRESS_IP} prometheus.local
${INGRESS_IP} minio.local
${INGRESS_IP} minio-console.local
${INGRESS_IP} feast.local
EOF

echo "DNS entries added successfully!"
echo "Verifying entries:"
cat /etc/hosts | grep local
