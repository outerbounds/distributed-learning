# TorchTune Distributed Training Example

This example demonstrates distributed training using Meta's TorchTune framework with Metaflow for orchestration. It implements a Supervised Fine-Tuning (SFT) recipe for large language models.

## Features

- Fully Sharded Data Parallel (FSDP) training
- Multi-node and multi-GPU training
- Checkpoint resumption for fault tolerance
- Activation checkpointing and offloading
- Gradient accumulation
- Preconfigured for H100 GPUs

## Directory Structure

```
torchtune/
├── full_finetuning/
│   ├── distributed_ft_recipe.py    # TorchTune recipe for fine-tuning
│   ├── h100_torchtune.py           # Metaflow flow definition
│   ├── metaflow_tune_checkpointer.py # Custom checkpointing for TorchTune
│   ├── metaflow_utils.py           # Utility functions
│   ├── multi_node_configs/         # Configuration files for multi-node training
│   └── single_node_configs/        # Configuration files for single-node training
```

## Running the Flow

The main flow is defined in `h100_torchtune.py`. You can run it with Metaflow as follows:

```bash
# From the torchtune/torchtune directory
python h100_torchtune.py --environment=fast-bakery run --config multi_node_configs/1b_config.yaml
```

## Configuration Files

The flow requires a configuration file that specifies training parameters. Example configurations are available in:

- `multi_node_configs/`: For multi-node training setups
- `single_node_configs/`: For single-node training setups: To run on single node Address the `# [CHANGE-FOR-SINGLE-NODE]` comments left in the [h100_torchtune.py](./torchtune/h100_torchtune.py)
