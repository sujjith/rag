# Phase 01: Foundations & Dev Environment

**Duration**: 2 weeks | **Prerequisites**: Basic programming knowledge

---

## Learning Objectives

By the end of this phase, you will:
- [ ] Set up a complete Python development environment
- [ ] Master `uv` for project and dependency management
- [ ] Work efficiently with Jupyter notebooks
- [ ] Configure VS Code for ML development

---

## Week 1: Python Environment Setup

### Day 1-2: Install uv Package Manager

**Why uv?** 10-100x faster than pip, built in Rust, replaces pip, pip-tools, virtualenv.

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Verify installation
uv --version

# Create new project
mkdir ~/ml-learning && cd ~/ml-learning
uv init

# Add dependencies
uv add numpy pandas scikit-learn matplotlib
```

**Hands-on Exercise:**
1. Create a new project called `ml-foundations`
2. Add `numpy`, `pandas`, `matplotlib`
3. Create a simple script that loads a CSV and plots data

---

### Day 3-4: JupyterLab Setup

```bash
# Install JupyterLab
uv add jupyterlab ipykernel

# Register kernel
uv run python -m ipykernel install --user --name=ml-foundations

# Launch JupyterLab
uv run jupyter lab
```

**Hands-on Exercise:**
1. Create notebook `01_data_exploration.ipynb`
2. Load a sample dataset (e.g., Iris)
3. Create basic visualizations
4. Practice markdown cells for documentation

---

### Day 5-7: VS Code Configuration

**Essential Extensions:**
| Extension | Purpose |
|-----------|---------|
| Python | Core Python support |
| Pylance | Fast type checking |
| Jupyter | Notebook support |
| GitLens | Git visualization |
| Error Lens | Inline error display |

**settings.json Configuration:**
```json
{
  "python.defaultInterpreterPath": ".venv/bin/python",
  "python.analysis.typeCheckingMode": "basic",
  "editor.formatOnSave": true,
  "python.formatting.provider": "black"
}
```

**Hands-on Exercise:**
1. Install all extensions
2. Configure settings.json
3. Open your project in VS Code
4. Run a notebook inside VS Code

---

## Week 2: Advanced Notebook Workflows

### Day 8-9: Marimo Reactive Notebooks

**Why Marimo?** Reactive execution, pure Python, git-friendly.

```bash
# Install Marimo
uv add marimo

# Create new notebook
uv run marimo edit my_notebook.py

# Convert from Jupyter
uv run marimo convert notebook.ipynb > notebook.py
```

**Hands-on Exercise:**
1. Create a Marimo notebook
2. Build an interactive data explorer with sliders
3. Compare workflow with Jupyter

---

### Day 10-11: Papermill for Notebook Automation

```bash
# Install Papermill
uv add papermill

# Parameterize notebook
uv run papermill input.ipynb output.ipynb -p param1 value1

# Run with different parameters
uv run papermill template.ipynb report_jan.ipynb -p month "January"
uv run papermill template.ipynb report_feb.ipynb -p month "February"
```

**Hands-on Exercise:**
1. Create parameterized notebook
2. Generate 3 reports with different parameters
3. Build a shell script to automate

---

### Day 12-14: Project Structure Best Practices

**Recommended ML Project Structure:**
```
ml-project/
├── pyproject.toml      # Project config (uv)
├── README.md
├── .gitignore
├── src/
│   └── ml_project/
│       ├── __init__.py
│       ├── data/
│       ├── features/
│       ├── models/
│       └── utils/
├── notebooks/
│   ├── 01_exploration.ipynb
│   └── 02_modeling.ipynb
├── tests/
├── data/
│   ├── raw/
│   └── processed/
└── configs/
```

**Hands-on Exercise:**
1. Create this structure
2. Add `.gitignore` for ML projects
3. Initialize git repository
4. Make first commit

---

## Milestone Checklist

- [ ] `uv` installed and working
- [ ] Created first project with `uv init`
- [ ] JupyterLab running locally
- [ ] VS Code configured with extensions
- [ ] Tried Marimo notebooks
- [ ] Automated notebook with Papermill
- [ ] Set up proper project structure
- [ ] First git commit made

---

## Resources

- [uv Documentation](https://docs.astral.sh/uv/)
- [JupyterLab Docs](https://jupyterlab.readthedocs.io/)
- [Marimo Docs](https://docs.marimo.io/)
- [Papermill Docs](https://papermill.readthedocs.io/)

---

**Next Phase**: [Phase 02 - ML Experiment Lifecycle](./phase_02_experiment_lifecycle.md)
