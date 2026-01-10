import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.datasets import load_iris
    return load_iris, mo, pd, plt


@app.cell
def _(mo):
    # Create a slider for sample size
    sample_size = mo.ui.slider(10, 150, value=50, label="Sample Size")
    sample_size
    return (sample_size,)


@app.cell
def _(load_iris, pd, sample_size):
    # Load iris dataset
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['species'] = iris.target_names[iris.target]

    # Sample based on slider value
    sampled_df = df.sample(n=sample_size.value, random_state=42)
    sampled_df.head()
    return (sampled_df,)


@app.cell
def _(plt, sample_size, sampled_df):
    # Plot updates when slider changes!
    plt.figure(figsize=(10, 6))
    plt.scatter(sampled_df['sepal length (cm)'], 
                sampled_df['sepal width (cm)'], 
                c=sampled_df['species'].map({'setosa': 0, 'versicolor': 1, 'virginica': 2}),
                cmap='viridis', alpha=0.6)
    plt.xlabel('Sepal Length (cm)')
    plt.ylabel('Sepal Width (cm)')
    plt.title(f'Iris Dataset Sample (n={sample_size.value})')
    plt.colorbar(label='Species')
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
 
    """)
    return


if __name__ == "__main__":
    app.run()
