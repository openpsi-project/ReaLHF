import multiprocessing
import os


def format_file(file_path):
    os.system(f"yapf -rip {file_path} && isort {file_path}")


if __name__ == "__main__":
    pool = multiprocessing.Pool()
    pool.map(format_file, ["api", "base", "impl", "scheduler", "apps", "experiments", "system"])
