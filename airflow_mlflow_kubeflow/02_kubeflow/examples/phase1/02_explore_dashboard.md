# Phase 1.2: Exploring the Kubeflow Dashboard

## Access the Dashboard

1. Start port forwarding:
   ```bash
   ./scripts/04_port_forward.sh
   ```

2. Open browser: http://localhost:8080

3. Login with:
   - Email: `user@example.com`
   - Password: `12341234`

## Dashboard Components

### 1. Home
- Overview of your namespace
- Quick links to common actions

### 2. Notebooks
- Create Jupyter notebook servers
- Manage running notebooks
- Access VSCode servers

### 3. Volumes
- Create persistent volumes
- Manage storage for notebooks and experiments

### 4. Pipelines
- Upload and manage pipelines
- Create and monitor runs
- View experiments

### 5. Experiments (AutoML)
- Katib hyperparameter tuning
- Neural architecture search

### 6. Models
- (KServe) Model serving management

## Exercise: Create Your First Notebook

1. Go to **Notebooks** in the sidebar
2. Click **+ New Notebook**
3. Configure:
   - Name: `my-first-notebook`
   - Image: `kubeflownotebookswg/jupyter-scipy:v1.8.0`
   - CPU: 0.5
   - Memory: 1Gi
4. Click **Launch**
5. Wait for the notebook to start (1-2 minutes)
6. Click **CONNECT** to open Jupyter

## Exercise: Explore the Pipelines UI

1. Go to **Pipelines** in the sidebar
2. Explore:
   - **Pipelines**: List of uploaded pipelines
   - **Experiments**: Groups of pipeline runs
   - **Runs**: Individual pipeline executions
   - **Recurring Runs**: Scheduled pipelines

## Kubernetes Resources

The dashboard creates these Kubernetes resources:

```bash
# List your notebooks
kubectl get notebooks -n kubeflow-user-example-com

# List your volumes
kubectl get pvc -n kubeflow-user-example-com

# List pipeline runs
kubectl get workflows -n kubeflow-user-example-com
```

## Next Steps

Once you're comfortable with the dashboard:
- Phase 1.3: Understanding Kubeflow concepts
- Phase 2: Building Kubeflow Pipelines
