import os
import re

from reallm.base.importing import import_module

# Import all dataset implementations.
_p = re.compile(r"^(?!.*__init__).*\.py$")
_filepath = os.path.dirname(__file__)
import_module(_filepath, _p)
