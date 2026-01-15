# uc_1_churn_prediction/tasks/data_versioning.py
"""Data versioning tasks using DVC."""

from prefect import task
import subprocess
import shutil
from pathlib import Path


@task(name="Version with DVC")
def version_with_dvc(file_path: str) -> str:
    """
    Step 3: Version dataset with DVC and push to MinIO.

    Args:
        file_path: Path to file to version

    Returns:
        str: Path to .dvc file
    """
    # Copy to data directory for versioning
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    dest_path = data_dir / Path(file_path).name
    shutil.copy(file_path, dest_path)

    # Add file to DVC
    subprocess.run(["dvc", "add", str(dest_path)], check=True)

    # Push to remote (MinIO)
    subprocess.run(["dvc", "push"], check=True)

    # Git commit the .dvc file
    dvc_file = f"{dest_path}.dvc"
    subprocess.run(["git", "add", dvc_file, ".gitignore"], check=True)
    subprocess.run(
        ["git", "commit", "-m", f"Add dataset version: {Path(file_path).name}"],
        check=True
    )

    print(f"Versioned: {file_path} -> {dvc_file}")
    return dvc_file
