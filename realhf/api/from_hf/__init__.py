import os
import re

from realhf.base.importing import import_module

# load_state_dict does not work on this model.
# The weights are not changed after loading.
# Skip this model temporarily.
import_module(os.path.dirname(__file__), re.compile(r"^(?!.*__init__).*\.py$"))
