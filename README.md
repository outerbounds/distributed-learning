# Multi-Node Training Examples

This repository contains examples of distributed training recipes using Metaflow for orchestration. The examples demonstrate different approaches to distributed training for large language models (LLMs) using frameworks like TorchTune and Hugging Face.

## Repository Structure

The repository is organized into separate examples, each showcasing a different distributed training approach:

- **TorchTune**: Implementation of supervised fine-tuning (SFT) using Meta's TorchTune framework.
- **Hugging Face GRPO**: Implementation of General Reward Prompting Optimization (GRPO) training using Hugging Face's TRL library.

## Getting Started

### Prerequisites

- Outerbounds and `metaflow-torchrun` installed (`pip install outerbounds metaflow-torchrun`)
- Kubernetes cluster with GPU nodes (examples are configured for H100 GPUs)
- Access to model weights (most examples pull from Hugging Face Hub)


## Examples Overview

### TorchTune (Supervised Fine-Tuning)

The TorchTune example demonstrates supervised fine-tuning for large language models with support for:
- Fully Sharded Data Parallel (FSDP) training
- Multi-node distributed training
- Support for checkpoint resumption
- Gradient accumulation
- Activation checkpointing and offloading

See the [TorchTune README](./torchtune/README.md) for more details.

### Hugging Face GRPO (General Reward Prompting Optimization)

The Hugging Face GRPO example demonstrates training with General Reward Prompting Optimization which:
- Uses reward functions like accuracy and formatting
- Leverages Hugging Face's TRL library
- Supports multi-node training with DeepSpeed
- Shows integration with various accelerator configurations

See the [Hugging Face GRPO README](./huggingface/grpo/README.md) for more details.