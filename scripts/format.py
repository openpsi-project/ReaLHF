from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import multiprocessing
import os
import subprocess


def format_pyfile(file_path):
    os.system(f"yapf -rip {file_path} && isort {file_path}")


def format_cppfile(file_path):
    subprocess.run(["clang-format", "-i", file_path])


if __name__ == "__main__":
    executor = ProcessPoolExecutor()
    for root, dirnames, files in os.walk(Path(__file__).parent.parent.absolute()):
        if "thirdparty" in root or "pycache" in root or ".git" in root or ".vscode" in root:
            continue
        for file in files:
            if file.endswith(".py"):
                executor.submit(format_pyfile, os.path.join(root, file))
            if file.endswith((".c", ".h", ".cpp", ".hpp", ".cu", ".cuh")):
                executor.submit(format_cppfile, os.path.join(root, file))
    executor.shutdown(wait=True)
