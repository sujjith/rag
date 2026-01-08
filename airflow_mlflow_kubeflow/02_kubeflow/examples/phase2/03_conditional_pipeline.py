"""
Phase 2.3: Conditional Pipeline

Demonstrates:
- Conditional execution
- Parallel tasks
- Pipeline parameters

Run:
  python 03_conditional_pipeline.py
"""
from kfp import dsl
from kfp import compiler


@dsl.component(base_image="python:3.11-slim")
def generate_random_number(min_val: int, max_val: int) -> int:
    """Generate a random number."""
    import random
    number = random.randint(min_val, max_val)
    print(f"Generated number: {number}")
    return number


@dsl.component(base_image="python:3.11-slim")
def check_threshold(number: int, threshold: int) -> str:
    """Check if number exceeds threshold."""
    result = "high" if number >= threshold else "low"
    print(f"Number {number} is {result} (threshold: {threshold})")
    return result


@dsl.component(base_image="python:3.11-slim")
def process_high_value(number: int) -> str:
    """Process high value numbers."""
    result = f"HIGH: Processing {number} with heavy computation"
    print(result)
    return result


@dsl.component(base_image="python:3.11-slim")
def process_low_value(number: int) -> str:
    """Process low value numbers."""
    result = f"LOW: Processing {number} with light computation"
    print(result)
    return result


@dsl.component(base_image="python:3.11-slim")
def summarize(results: list) -> str:
    """Summarize all results."""
    summary = f"Processed {len(results)} results"
    for r in results:
        print(f"  - {r}")
    return summary


@dsl.pipeline(
    name="conditional-pipeline",
    description="Pipeline with conditional execution and parallel tasks"
)
def conditional_pipeline(
    num_iterations: int = 3,
    threshold: int = 50
):
    """
    Pipeline that:
    1. Generates random numbers
    2. Conditionally processes based on threshold
    """
    with dsl.ParallelFor(items=list(range(num_iterations))) as iteration:
        # Generate random number
        gen_task = generate_random_number(min_val=0, max_val=100)

        # Check threshold
        check_task = check_threshold(
            number=gen_task.output,
            threshold=threshold
        )

        # Conditional processing
        with dsl.If(check_task.output == "high"):
            high_task = process_high_value(number=gen_task.output)

        with dsl.If(check_task.output == "low"):
            low_task = process_low_value(number=gen_task.output)


if __name__ == "__main__":
    output_file = "conditional_pipeline.yaml"

    compiler.Compiler().compile(
        pipeline_func=conditional_pipeline,
        package_path=output_file
    )

    print("=" * 50)
    print("Conditional Pipeline compiled!")
    print("=" * 50)
    print(f"\nOutput: {output_file}")
    print("\nFeatures demonstrated:")
    print("  - ParallelFor loops")
    print("  - Conditional execution (dsl.If)")
    print("  - Dynamic task creation")
    print("=" * 50)
