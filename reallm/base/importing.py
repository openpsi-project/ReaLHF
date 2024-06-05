from pathlib import Path
import importlib
import os
import re
import importlib.util
import sys


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

def import_usercode(module_path: str, module_name: str):
    # Create a module spec
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    # Create a module object
    module = importlib.util.module_from_spec(spec)
    # Add the module to sys.modules
    sys.modules[module_name] = module
    # Execute the module in its own namespace
    spec.loader.exec_module(module)
