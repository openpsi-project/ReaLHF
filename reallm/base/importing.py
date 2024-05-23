from pathlib import Path
import importlib
import os
import re


def import_module(path: str, pattern: re.Pattern):
    dirname = Path(path)
    for x in os.listdir(dirname.absolute()):
        if not pattern.match(x):
            continue
        module_path = os.path.splitext(os.path.join(dirname, x))[0]
        assert "reallm" in module_path
        start_idx = path.index("reallm")
        module_path = module_path[start_idx:]
        module_path = module_path.replace(os.sep, ".")
        importlib.import_module(module_path)
