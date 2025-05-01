# Hugging Face GRPO Training Example

This example demonstrates distributed training using Hugging Face's General Reward Prompting Optimization (GRPO) framework with Metaflow for orchestration. GRPO is a training approach that leverages human feedback through reward modeling.

## Features

- Multi-node and multi-GPU training with DeepSpeed Zero-3
- Distributed training with Accelerate
- Integrated reward functions for accuracy and format verification
- Model downloading from Hugging Face Hub
- Preconfigured for H100 GPUs

## Directory Structure

```
huggingface/grpo/
├── h100_grpo.py           # Metaflow flow definition
├── grpo.py                # GRPO training implementation
├── infa_configs.py        # Infrastructure configuration
├── metaflow_utils.py      # Utility functions
├── config.yaml            # Default training configuration
└── accelerate_configs/    # Accelerate configuration files for distributed training
```

## Running the Flow

The main flow is defined in `h100_grpo.py`. You can run it with Metaflow as follows:

```bash
# From the huggingface/grpo directory
python h100_grpo.py --environment=fast-bakery run --config config.yaml
```


To specify a different accelerate config you can run the following :

```bash
python h100_grpo.py --config accelerate_config ./accelerate_configs/zero3.yaml --environment=fast-bakery run --config config.yaml # you can --dry-run for minimal configurations
```

## Configuration Files

### Training Configuration (config.yaml)

The config file is composed of multiple configs like the [GRPOConfig](https://github.com/huggingface/trl/blob/main/trl/trainer/grpo_config.py) and [ModelConfig](https://github.com/huggingface/trl/blob/main/trl/trainer/model_config.py)

### Accelerate Configuration

Accelerate configs are stored in `accelerate_configs/` and determine how distributed training is set up. You can use your own custom config or use any examples configs from accelerate [like here](https://github.com/huggingface/trl/tree/main/examples/accelerate_configs).

## Reward Functions

The example includes two reward functions:
- `accuracy_reward`: Checks if the answer matches the ground truth
- `format_reward`: Ensures the response follows the required format with thinking and answer sections
