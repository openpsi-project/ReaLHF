# Pipeline Parallel
Pipeline parallel in the system follows the implementation of DeepSpeed [PipelineEngine](https://github.com/microsoft/DeepSpeed/blob/master/deepspeed/runtime/pipe/engine.py), supports GPipe and 1F1B style pipeline parallel training and generating. We only support pipeline parallel for FlashMQAT models.


## Preparing Checkpoints
Before running pipeline parallel experiments, you must transform your huggingface-style model checkpoints into sharded pipeline parallel checkpoints with [this script](../../scripts/transform_to_pipe_ckpt.py). This script automatically partitions model checkpoints into several files. One file can only contains parameters from **one pipeline stage**. Parameters from the same stage can be further splitted into smaller shards to reduce memory usage when loading models (which can be done by setting `NUM_SHARDS`>1 in the script). **NOTE: When running experiments, you should match your num_pipeline_stages setting with your pipeline checkpoint files. If you need to change num_pipeline_stages, please rerun the script for new pipeline checkpoint files!** 


## [DeepSpeedPipelineEngine](../../impl/model/backend/pipe_engine/ds_pipe_engine.py)
`DeepSpeedPipelineEngine` provides 4 main APIs for train, evaluation, generation and forward for interfaces:

1. `train_batch(packed_input_ids, cu_seqlens, loss_fn, **loss_fn_kwargs)`
2. `eval_batch(packed_input_ids, cu_seqlens, loss_fn, **loss_fn_kwargs)`
3. `generate(packed_input_ids, cu_seqlens, tokenizer, gconfig)`
4. `forward(packed_input_ids, cu_seqlens)`

All these APIs takes packed inputs, separate them into micro batches and execute instructions on micro batches in a pipeline. They only return results (losses, training stats, generated tokens etc.) on the last pipeline stage. On other stages these APIs return None, which should be dealt with in interface implementations. 

Other APIs:
1. `train()`: similar to pytorch module.train()
2. `eval()`: similar to pytorch module.eval()
3. `set_version_steps(version_steps)`: Set `version_steps`. `version_steps` are used as lr_kwargs passed into optimizer steps, see Pipe PPO Interface for an example.

Notes: 
1. You cannot directly call `backward()` and `step()` of `DeepSpeedPipelineEngine` like original `DeepSpeedEngine` because these single operations are meaningless in pipeline parallel.
2. The only configurable option in the engine is num_micro_batches. In default, num_micro_batches == num_pipeline_stages, and we need to always keep num_micro_batches >= num_pipeline_stages for good performance.
3. The inputs for 3 main APIs are separated into micro batches using  `base.dataparallel.PackedParallelDataBroker`. This means that micro batches in pipeline parallel are splitted evenly by tokens instead of numbers of sequences, and micro batches could contain different numbers of sequences. 
4. `loss_fn` argument in `train_batch` and `eval_batch` should be implemented for each interface. In pipeline engine, losses are calculated separately for each micro batch. As a result, the `loss_fn_kwargs` should take two types of inputs. One type is packed tensors (which should be able to be splitted into function inputs of micro batches by [`base.dataparallel.PackedParallelDataBroker`](../../base/dataparallel.py)). The other type is non-tensor constants, which is copied across micro batches.
5. The outputs of `loss_fn` should be a tuple of **loss** (a tensor of single value) and **stats** (stats to be recorded). The outputs of `train_batch` and `eval_batch` are tuples of **average loss** and **list of stats** of micro batches. There are examples of `loss_fn` implementations in example interfaces.
6. In `train_batch`, optimization and data parallel gradient all reduce happens once after all micro batches finish forward and backward passes.
7. In `forward`, the output is packed logits of all micro batches.

## Interface Examples
Interface logics are identical to [flash interfaces](../../impl/model/interface/flash), but implemented with `DeepSpeedPipelineEngine` APIs:
1. [Pipe SFT Interface](../../impl/model/interface/pipe/pipe_sft_flash_interface.py)
2. [Pipe PPO Interface](../../impl/model/interface/pipe/pipe_ppo_flash_interface.py)

Note:
1. Training stats in the interface are not reduced across data parallel ranks. Both actor and critic training stats will be printed in the interface per data parallel rank.

## Example Experiments
1. [Pipe SFT Experiments](../../experiments/wpsf_sft_pipe.py)
2. [Pipe PPO Experiments](../../experiments/wpsf_ppo_pipe.py) 

Notes:
1. Do not set num_pipeline_parallel to 1
2. In ModelBackend, pipeline parallel does not support `zero_stage > 1`.
3. Usually only actor models requires pipeline parallel, but critic models can use pipeline parallel if the batch size or context length is large. Use PPO critic interface in [Pipe PPO Interface](../../impl/model/interface/pipe/pipe_ppo_flash_interface.py) and critic models in [pipe_nn.py](../../impl/model/nn/pipe_nn.py) to enable pipeline parallel for critic models.
4. [This issue](https://github.com/garrett4wade/distributed_llm/issues/56) can cause error when training. Setting **(batch size per device / pipeline_stage) > 4** might fix this issue. 
