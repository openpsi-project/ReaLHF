from pathlib import Path
import importlib
import os

# Import all existing HuggingFace model registries.
hf_impl_path = Path(os.path.dirname(__file__))
for x in os.listdir(hf_impl_path.absolute()):
    if not x.endswith(".py"):
        continue
    if "__init__" in x:
        continue
    if "starcoder" in x.strip('.py'):
        # HACK: StarCoder seems to have a bug with transformers v0.39.1:
        # load_state_dict does not work on this model,
        # and the weights are not changed after loading. Skip this model temporarily.
        continue
    importlib.import_module(f"reallm.api.from_hf.{x.strip('.py')}")