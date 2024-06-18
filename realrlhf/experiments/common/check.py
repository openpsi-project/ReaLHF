import realrlhf.api.core.model_api as model_api


def check_is_realrlhf_native_impl(_cls):
    return _cls.__module__.startswith("realrlhf")


def check_is_realrlhf_native_model_interface(name):
    import realrlhf.impl.model.interface.dpo_interface
    import realrlhf.impl.model.interface.ppo_interface
    import realrlhf.impl.model.interface.rw_interface
    import realrlhf.impl.model.interface.sft_interface

    _cls = model_api.ALL_INTERFACE_CLASSES.get(name)
    return _cls and check_is_realrlhf_native_impl(_cls)
