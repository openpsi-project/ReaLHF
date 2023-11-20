# Code Style

We use `yapf` and `isort` to format code files.
Install them via `pip install -U yapf isort`.
See `pyproject.toml` for detailed configurations.

## Formatting
Run ```yapf -ri .``` or ```./format all``` in the repository root.

## Import Sorting
Run ```isort .``` in the repository root.

## Before Pull Request
Please run `yapf -ri . && isort .` before opening a pull request.