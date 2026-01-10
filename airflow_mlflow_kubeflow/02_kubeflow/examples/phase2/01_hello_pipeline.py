"""
Phase 2.1: Hello World Pipeline

Your first Kubeflow Pipeline demonstrating basic concepts:
- Components
- Pipeline definition
- Compilation

Run:
  python 01_hello_pipeline.py
  # Then upload hello_pipeline.yaml to Kubeflow UI
"""
from kfp import dsl
from kfp import compiler


# Component 1: Say hello
@dsl.component(base_image="python:3.11-slim")
def say_hello(name: str) -> str:
    """A component that says hello."""
    message = f"Hello, {name}!"
    print(message)
    return message


# Component 2: Make it uppercase
@dsl.component(base_image="python:3.11-slim")
def to_uppercase(text: str) -> str:
    """Convert text to uppercase."""
    result = text.upper()
    print(f"Uppercase: {result}")
    return result


# Component 3: Add exclamation
@dsl.component(base_image="python:3.11-slim")
def add_emphasis(text: str, num_exclamations: int = 3) -> str:
    """Add exclamation marks."""
    result = text + "!" * num_exclamations
    print(f"Final: {result}")
    return result


# Pipeline definition
@dsl.pipeline(
    name="hello-world-pipeline",
    description="A simple hello world pipeline to demonstrate KFP basics"
)
def hello_pipeline(name: str = "Kubeflow", exclamations: int = 3):
    """
    A simple pipeline that:
    1. Says hello to a name
    2. Converts to uppercase
    3. Adds exclamation marks
    """
    # Task 1: Say hello
    hello_task = say_hello(name=name)

    # Task 2: Convert to uppercase (depends on task 1)
    upper_task = to_uppercase(text=hello_task.output)

    # Task 3: Add emphasis (depends on task 2)
    emphasis_task = add_emphasis(
        text=upper_task.output,
        num_exclamations=exclamations
    )


# Compile the pipeline
if __name__ == "__main__":
    output_file = "hello_pipeline.yaml"

    compiler.Compiler().compile(
        pipeline_func=hello_pipeline,
        package_path=output_file
    )

    print("=" * 50)
    print("Pipeline compiled successfully!")
    print("=" * 50)
    print(f"\nOutput: {output_file}")
    print("\nTo run this pipeline:")
    print("1. Open Kubeflow Dashboard: http://localhost:8080")
    print("2. Go to Pipelines → Upload Pipeline")
    print(f"3. Upload {output_file}")
    print("4. Create a Run")
    print("=" * 50)
