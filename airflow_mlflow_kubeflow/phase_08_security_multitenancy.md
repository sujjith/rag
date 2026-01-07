# Phase 08: Security & Multi-tenancy

## Overview

Enterprise security setup including RBAC, OAuth/OIDC authentication, secrets management, network policies, and multi-tenancy isolation.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        SECURITY ARCHITECTURE                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                      IDENTITY PROVIDER                               │   │
│   │                  (Keycloak / Okta / Azure AD)                        │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                     │                                        │
│                                     │ OAuth/OIDC                            │
│                                     ▼                                        │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                      ISTIO SERVICE MESH                              │   │
│   │                  (mTLS, AuthZ, Rate Limiting)                        │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                     │                                        │
│          ┌──────────────────────────┼──────────────────────────┐            │
│          │                          │                          │            │
│          ▼                          ▼                          ▼            │
│   ┌─────────────┐           ┌─────────────┐           ┌─────────────┐      │
│   │   Team A    │           │   Team B    │           │   Team C    │      │
│   │ Namespace   │           │ Namespace   │           │ Namespace   │      │
│   │  ┌───────┐  │           │  ┌───────┐  │           │  ┌───────┐  │      │
│   │  │ RBAC  │  │           │  │ RBAC  │  │           │  │ RBAC  │  │      │
│   │  └───────┘  │           │  └───────┘  │           │  └───────┘  │      │
│   │  ┌───────┐  │           │  ┌───────┐  │           │  ┌───────┐  │      │
│   │  │Secrets│  │           │  │Secrets│  │           │  │Secrets│  │      │
│   │  └───────┘  │           │  └───────┘  │           │  └───────┘  │      │
│   └─────────────┘           └─────────────┘           └─────────────┘      │
│                                                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                     HASHICORP VAULT                                  │   │
│   │                   (Secrets Management)                               │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Step 1: Install Keycloak (Identity Provider)

Create `security/keycloak/values.yaml`:

```yaml
auth:
  adminUser: admin
  adminPassword: admin123

postgresql:
  enabled: true
  auth:
    postgresPassword: keycloak123
    database: keycloak

production: false

proxy: edge

ingress:
  enabled: true
  ingressClassName: nginx
  hostname: keycloak.local
  tls: true

extraEnvVars:
  - name: KEYCLOAK_EXTRA_ARGS
    value: "--spi-login-protocol-openid-connect-legacy-logout-redirect-uri=true"
```

```bash
# Add Bitnami repo
helm repo add bitnami https://charts.bitnami.com/bitnami

# Install Keycloak
helm install keycloak bitnami/keycloak \
    --namespace security \
    --create-namespace \
    --values security/keycloak/values.yaml
```

### Configure Keycloak Realm

```bash
# Access Keycloak admin console: https://keycloak.local
# Create realm: mlops-platform

# Create clients:
# - airflow-client (confidential)
# - mlflow-client (confidential)
# - kubeflow-client (confidential)
# - grafana-client (confidential)

# Create roles:
# - ml-admin
# - ml-engineer
# - data-scientist
# - viewer

# Create groups:
# - team-a
# - team-b
# - platform-admin
```

---

## Step 2: Install HashiCorp Vault

Create `security/vault/values.yaml`:

```yaml
server:
  ha:
    enabled: true
    replicas: 3
    raft:
      enabled: true
      config: |
        ui = true

        listener "tcp" {
          tls_disable = 1
          address = "[::]:8200"
          cluster_address = "[::]:8201"
        }

        storage "raft" {
          path = "/vault/data"
        }

        service_registration "kubernetes" {}

  dataStorage:
    enabled: true
    size: 10Gi

  ingress:
    enabled: true
    ingressClassName: nginx
    hosts:
      - host: vault.local

injector:
  enabled: true

ui:
  enabled: true
```

```bash
# Add HashiCorp repo
helm repo add hashicorp https://helm.releases.hashicorp.com

# Install Vault
helm install vault hashicorp/vault \
    --namespace security \
    --values security/vault/values.yaml

# Initialize Vault
kubectl exec -n security vault-0 -- vault operator init

# Unseal Vault (use 3 of 5 keys)
kubectl exec -n security vault-0 -- vault operator unseal <key1>
kubectl exec -n security vault-0 -- vault operator unseal <key2>
kubectl exec -n security vault-0 -- vault operator unseal <key3>
```

### Configure Vault

```bash
# Login to Vault
kubectl exec -it -n security vault-0 -- vault login

# Enable Kubernetes auth
kubectl exec -n security vault-0 -- vault auth enable kubernetes

# Configure Kubernetes auth
kubectl exec -n security vault-0 -- vault write auth/kubernetes/config \
    kubernetes_host="https://$KUBERNETES_PORT_443_TCP_ADDR:443"

# Create secrets engine
kubectl exec -n security vault-0 -- vault secrets enable -path=mlops kv-v2

# Store secrets
kubectl exec -n security vault-0 -- vault kv put mlops/database \
    postgres_password=postgres123 \
    mlflow_password=mlflow123 \
    airflow_password=airflow123

kubectl exec -n security vault-0 -- vault kv put mlops/minio \
    access_key=minio \
    secret_key=minio123

kubectl exec -n security vault-0 -- vault kv put mlops/api-keys \
    slack_webhook=https://hooks.slack.com/... \
    pagerduty_key=...
```

### Create Vault Policies

```hcl
# security/vault/policies/ml-engineer.hcl
path "mlops/data/database" {
  capabilities = ["read"]
}

path "mlops/data/minio" {
  capabilities = ["read"]
}

path "mlops/data/api-keys" {
  capabilities = ["deny"]
}

# security/vault/policies/ml-admin.hcl
path "mlops/*" {
  capabilities = ["create", "read", "update", "delete", "list"]
}
```

```bash
# Apply policies
kubectl exec -n security vault-0 -- vault policy write ml-engineer /vault/policies/ml-engineer.hcl
kubectl exec -n security vault-0 -- vault policy write ml-admin /vault/policies/ml-admin.hcl

# Create roles for Kubernetes service accounts
kubectl exec -n security vault-0 -- vault write auth/kubernetes/role/airflow \
    bound_service_account_names=airflow \
    bound_service_account_namespaces=airflow \
    policies=ml-engineer \
    ttl=1h

kubectl exec -n security vault-0 -- vault write auth/kubernetes/role/mlflow \
    bound_service_account_names=mlflow \
    bound_service_account_namespaces=mlflow \
    policies=ml-engineer \
    ttl=1h
```

---

## Step 3: Kubernetes RBAC

### Namespace-Based Multi-tenancy

```yaml
# security/rbac/namespaces.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: team-a
  labels:
    team: team-a
    istio-injection: enabled
---
apiVersion: v1
kind: Namespace
metadata:
  name: team-b
  labels:
    team: team-b
    istio-injection: enabled
```

### RBAC Roles

```yaml
# security/rbac/roles.yaml
---
# ML Engineer Role
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: ml-engineer
  namespace: team-a
rules:
  - apiGroups: [""]
    resources: ["pods", "pods/log", "services", "configmaps", "secrets"]
    verbs: ["get", "list", "watch", "create", "update", "delete"]
  - apiGroups: ["apps"]
    resources: ["deployments", "statefulsets"]
    verbs: ["get", "list", "watch", "create", "update", "delete"]
  - apiGroups: ["batch"]
    resources: ["jobs", "cronjobs"]
    verbs: ["get", "list", "watch", "create", "update", "delete"]
  - apiGroups: ["kubeflow.org"]
    resources: ["notebooks", "tfjobs", "pytorchjobs"]
    verbs: ["get", "list", "watch", "create", "update", "delete"]
  - apiGroups: ["serving.kserve.io"]
    resources: ["inferenceservices"]
    verbs: ["get", "list", "watch", "create", "update", "delete"]
---
# Data Scientist Role (more restricted)
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: data-scientist
  namespace: team-a
rules:
  - apiGroups: [""]
    resources: ["pods", "pods/log", "configmaps"]
    verbs: ["get", "list", "watch"]
  - apiGroups: ["kubeflow.org"]
    resources: ["notebooks"]
    verbs: ["get", "list", "watch", "create", "update", "delete"]
  - apiGroups: ["kubeflow.org"]
    resources: ["tfjobs", "pytorchjobs"]
    verbs: ["get", "list", "watch", "create"]
---
# Viewer Role
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: viewer
  namespace: team-a
rules:
  - apiGroups: [""]
    resources: ["pods", "pods/log", "services", "configmaps"]
    verbs: ["get", "list", "watch"]
  - apiGroups: ["apps"]
    resources: ["deployments", "statefulsets"]
    verbs: ["get", "list", "watch"]
```

### Role Bindings

```yaml
# security/rbac/bindings.yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: ml-engineer-binding
  namespace: team-a
subjects:
  - kind: Group
    name: ml-engineers
    apiGroup: rbac.authorization.k8s.io
roleRef:
  kind: Role
  name: ml-engineer
  apiGroup: rbac.authorization.k8s.io
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: data-scientist-binding
  namespace: team-a
subjects:
  - kind: Group
    name: data-scientists
    apiGroup: rbac.authorization.k8s.io
roleRef:
  kind: Role
  name: data-scientist
  apiGroup: rbac.authorization.k8s.io
```

---

## Step 4: Network Policies

```yaml
# security/network-policies/default-deny.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: default-deny-all
  namespace: team-a
spec:
  podSelector: {}
  policyTypes:
    - Ingress
    - Egress
---
# Allow DNS
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: allow-dns
  namespace: team-a
spec:
  podSelector: {}
  policyTypes:
    - Egress
  egress:
    - to:
        - namespaceSelector:
            matchLabels:
              kubernetes.io/metadata.name: kube-system
      ports:
        - protocol: UDP
          port: 53
---
# Allow access to MLflow
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: allow-mlflow
  namespace: team-a
spec:
  podSelector: {}
  policyTypes:
    - Egress
  egress:
    - to:
        - namespaceSelector:
            matchLabels:
              kubernetes.io/metadata.name: mlflow
      ports:
        - protocol: TCP
          port: 5000
---
# Allow access to Feast
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: allow-feast
  namespace: team-a
spec:
  podSelector: {}
  policyTypes:
    - Egress
  egress:
    - to:
        - namespaceSelector:
            matchLabels:
              kubernetes.io/metadata.name: feast
      ports:
        - protocol: TCP
          port: 6566
---
# Allow ingress from Istio gateway
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: allow-istio-ingress
  namespace: team-a
spec:
  podSelector: {}
  policyTypes:
    - Ingress
  ingress:
    - from:
        - namespaceSelector:
            matchLabels:
              kubernetes.io/metadata.name: istio-system
```

---

## Step 5: Istio Security

### mTLS Configuration

```yaml
# security/istio/mtls.yaml
apiVersion: security.istio.io/v1beta1
kind: PeerAuthentication
metadata:
  name: default
  namespace: istio-system
spec:
  mtls:
    mode: STRICT
---
apiVersion: security.istio.io/v1beta1
kind: PeerAuthentication
metadata:
  name: team-a-mtls
  namespace: team-a
spec:
  mtls:
    mode: STRICT
```

### Authorization Policies

```yaml
# security/istio/authz.yaml
apiVersion: security.istio.io/v1beta1
kind: AuthorizationPolicy
metadata:
  name: deny-all
  namespace: team-a
spec:
  {}
---
apiVersion: security.istio.io/v1beta1
kind: AuthorizationPolicy
metadata:
  name: allow-ml-services
  namespace: team-a
spec:
  selector:
    matchLabels:
      app: ml-service
  rules:
    - from:
        - source:
            principals:
              - "cluster.local/ns/team-a/sa/ml-engineer"
              - "cluster.local/ns/airflow/sa/airflow"
      to:
        - operation:
            methods: ["GET", "POST"]
            paths: ["/predict", "/health"]
---
# JWT Authentication
apiVersion: security.istio.io/v1beta1
kind: RequestAuthentication
metadata:
  name: jwt-auth
  namespace: team-a
spec:
  selector:
    matchLabels:
      app: ml-service
  jwtRules:
    - issuer: "https://keycloak.local/realms/mlops-platform"
      jwksUri: "https://keycloak.local/realms/mlops-platform/protocol/openid-connect/certs"
      audiences:
        - "ml-services"
```

---

## Step 6: OAuth Integration for Services

### Airflow OAuth

```python
# airflow/webserver_config.py
from flask_appbuilder.security.manager import AUTH_OAUTH
import os

AUTH_TYPE = AUTH_OAUTH
AUTH_USER_REGISTRATION = True
AUTH_USER_REGISTRATION_ROLE = "Viewer"

OAUTH_PROVIDERS = [
    {
        'name': 'keycloak',
        'token_key': 'access_token',
        'icon': 'fa-key',
        'remote_app': {
            'client_id': os.getenv('KEYCLOAK_CLIENT_ID'),
            'client_secret': os.getenv('KEYCLOAK_CLIENT_SECRET'),
            'api_base_url': 'https://keycloak.local/realms/mlops-platform/protocol/openid-connect/',
            'client_kwargs': {
                'scope': 'openid email profile'
            },
            'access_token_url': 'https://keycloak.local/realms/mlops-platform/protocol/openid-connect/token',
            'authorize_url': 'https://keycloak.local/realms/mlops-platform/protocol/openid-connect/auth',
            'request_token_url': None,
        }
    }
]

# Map Keycloak roles to Airflow roles
AUTH_ROLES_MAPPING = {
    "ml-admin": ["Admin"],
    "ml-engineer": ["User", "Op"],
    "data-scientist": ["User"],
    "viewer": ["Viewer"],
}

AUTH_ROLES_SYNC_AT_LOGIN = True
```

### Grafana OAuth

```yaml
# In grafana values
grafana.ini:
  server:
    root_url: https://grafana.local
  auth.generic_oauth:
    enabled: true
    name: Keycloak
    allow_sign_up: true
    client_id: grafana-client
    client_secret: ${GRAFANA_OAUTH_SECRET}
    scopes: openid email profile
    auth_url: https://keycloak.local/realms/mlops-platform/protocol/openid-connect/auth
    token_url: https://keycloak.local/realms/mlops-platform/protocol/openid-connect/token
    api_url: https://keycloak.local/realms/mlops-platform/protocol/openid-connect/userinfo
    role_attribute_path: contains(groups[*], 'ml-admin') && 'Admin' || contains(groups[*], 'ml-engineer') && 'Editor' || 'Viewer'
```

---

## Step 7: Secrets Injection with Vault

### Vault Agent Injector

```yaml
# security/vault/injection-example.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-service
  namespace: team-a
spec:
  template:
    metadata:
      annotations:
        vault.hashicorp.com/agent-inject: "true"
        vault.hashicorp.com/role: "ml-engineer"
        vault.hashicorp.com/agent-inject-secret-database: "mlops/data/database"
        vault.hashicorp.com/agent-inject-template-database: |
          {{- with secret "mlops/data/database" -}}
          export POSTGRES_PASSWORD="{{ .Data.data.postgres_password }}"
          export MLFLOW_PASSWORD="{{ .Data.data.mlflow_password }}"
          {{- end -}}
        vault.hashicorp.com/agent-inject-secret-minio: "mlops/data/minio"
        vault.hashicorp.com/agent-inject-template-minio: |
          {{- with secret "mlops/data/minio" -}}
          export AWS_ACCESS_KEY_ID="{{ .Data.data.access_key }}"
          export AWS_SECRET_ACCESS_KEY="{{ .Data.data.secret_key }}"
          {{- end -}}
    spec:
      serviceAccountName: ml-engineer
      containers:
        - name: ml-service
          image: ml-service:latest
          command: ["/bin/sh", "-c"]
          args:
            - source /vault/secrets/database && source /vault/secrets/minio && python app.py
```

---

## Step 8: Resource Quotas

```yaml
# security/quotas/team-a-quota.yaml
apiVersion: v1
kind: ResourceQuota
metadata:
  name: team-a-quota
  namespace: team-a
spec:
  hard:
    requests.cpu: "20"
    requests.memory: 40Gi
    limits.cpu: "40"
    limits.memory: 80Gi
    requests.nvidia.com/gpu: "4"
    pods: "50"
    services: "20"
    persistentvolumeclaims: "20"
    requests.storage: 500Gi
---
apiVersion: v1
kind: LimitRange
metadata:
  name: team-a-limits
  namespace: team-a
spec:
  limits:
    - default:
        cpu: "1"
        memory: 2Gi
      defaultRequest:
        cpu: "500m"
        memory: 1Gi
      max:
        cpu: "8"
        memory: 32Gi
        nvidia.com/gpu: "2"
      min:
        cpu: "100m"
        memory: 128Mi
      type: Container
```

---

## Verification

```bash
#!/bin/bash
# verify_security.sh

echo "=== Security Verification ==="

echo -e "\n1. Keycloak Status:"
kubectl get pods -n security -l app.kubernetes.io/name=keycloak

echo -e "\n2. Vault Status:"
kubectl get pods -n security -l app.kubernetes.io/name=vault

echo -e "\n3. Network Policies:"
kubectl get networkpolicies -A

echo -e "\n4. RBAC Roles:"
kubectl get roles -A | grep -E "ml-|data-"

echo -e "\n5. Istio mTLS Status:"
istioctl analyze -A

echo -e "\n6. Resource Quotas:"
kubectl get resourcequota -A

echo -e "\n=== Verification Complete ==="
```

---

**Status**: Phase 08 Complete
**Features Covered**: RBAC, OAuth, Vault, Network Policies, mTLS, Multi-tenancy
