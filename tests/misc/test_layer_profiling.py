import argparse

from reallm.apps.main import main_profile_layers
from reallm.base.testing import MODEL_FAMILY_TO_PATH


def profile_testing_model_types():
    try:
        for model_family, model_path in MODEL_FAMILY_TO_PATH.items():
            args = argparse.Namespace(
                model_class=model_family._class,
                model_size=model_family.size,
                is_critic=model_family.is_critic,
                model_path=model_path,
            )
            main_profile_layers(args)
    except KeyboardInterrupt:
        return


if __name__ == "__main__":
    profile_testing_model_types()
