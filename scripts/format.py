from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import multiprocessing
import os
import subprocess


def format_pyfile(file_path):
    os.system(f"yapf -rip {file_path} && isort {file_path}")


def format_cppfile(file_path):
    subprocess.run(["clang-format", "-i", file_path])


def count_loc(file_path):
    return int(subprocess.check_output(f"wc -l {file_path}", shell=True).decode('ascii').split()[0])


if __name__ == "__main__":
    executor = ProcessPoolExecutor()
    python_loc = cpp_loc = 0
    for root, dirnames, files in os.walk(Path(__file__).parent.parent.absolute()):
        if "thirdparty" in root or "pycache" in root or ".git" in root or ".vscode" in root:
            continue
        for file in files:
            if file.endswith(".py"):
                python_loc += count_loc(os.path.join(root, file))
                executor.submit(format_pyfile, os.path.join(root, file))
            if file.endswith((".c", ".h", ".cpp", ".hpp", ".cu", ".cuh")):
                cpp_loc += count_loc(os.path.join(root, file))
                executor.submit(format_cppfile, os.path.join(root, file))
    executor.shutdown(wait=True)
    print(f"Formatted {python_loc} lines of Python code and {cpp_loc} lines of C++ code.")
