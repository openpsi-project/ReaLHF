import reallm.api.core.model_api as model_api


def check_is_reallm_native_impl(_cls):
    return _cls.__module__.startswith("reallm")


def check_is_reallm_native_model_interface(name):
    _cls = model_api.ALL_INTERFACE_CLASSES.get(name)
    return _cls and check_is_reallm_native_impl(_cls)
