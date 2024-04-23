import dataclasses
from experiments.autoexp.device_mapping import auto_device_mapping as auto
from experiments.common.ppo_exp import PPOHyperparameters
from api.config.dfg import ModelRPC as ModelFunctionCallDef
from api.config.dfg import ModelInterfaceType
from typing import List
GENERATE = ModelInterfaceType.GENERATE
INFERENCE = ModelInterfaceType.INFERENCE
TRAIN_STEP = ModelInterfaceType.TRAIN_STEP

# auto is a decorator that generates worker
# scheduling configs in the cluster.
@auto(nodelist="com[01-08]", batch_size=256)
@dataclasses.dataclass
class Experiment:
    seed: int = 1
    ppo: PPOHyperparameters

    @property
    def rpcs(self) -> List[ModelFunctionCallDef]:
        return [
            ModelFunctionCallDef(
                model_name="actor",
                model_type="llama7b",
                interface_type=GENERATE,
                input_data=["prompts"],
                output_data=["seq", "logp"],
            ),
            ModelFunctionCallDef(
                model_name="reward",
                model_type="llama7b-critic",
                interface_type=INFERENCE,
                input_data=["seq"],
                output_data=["r"],
            ),
            ModelFunctionCallDef(
                model_name="actor",
                interface_type=TRAIN_STEP,
                input_data=["seq", "r", ...],
            ),
            # ref inference, critic inference,
            # and critic training
            ...,
        ]