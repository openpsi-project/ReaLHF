import realhf.api.core.model_api as model_api


def check_is_realhf_native_impl(_cls):
    return _cls.__module__.startswith("realhf")


def check_is_realhf_native_model_interface(name):
    # NOTE: we should not use auto-importing here,
    # because the user may write customized interfaces under this folder.
    import realhf.impl.model.interface.dpo_interface
    import realhf.impl.model.interface.gen_interface
    import realhf.impl.model.interface.ppo_interface
    import realhf.impl.model.interface.rw_interface
    import realhf.impl.model.interface.sft_interface

    _cls = model_api.ALL_INTERFACE_CLASSES.get(name)
    return _cls and check_is_realhf_native_impl(_cls)
