from metaflow import (
    FlowSpec,
    step,
    current,
    Parameter,
    Config,
    secrets,
    kubernetes,
    pypi,
    card,
    gpu_profile,
    model,
    environment,
    IncludeFile,
    torchrun,
    project,
    huggingface_hub,
)
from metaflow_utils import Accelerate
from infa_configs import H100_K8S_CONFIG

DRY_RUN_PARAMS = dict(
    num_train_epochs=1,  # or even 0.5 if you use floats
    max_steps=20,  # Only 1 step
    per_device_train_batch_size=1,  # Small batch size
    logging_steps=1,
)


def huggingface(func):
    deco_list = [
        pypi(
            python="3.11.5",
            packages={
                "huggingface-hub[hf_transfer]": "0.25.2"
            },  # Installing Hugging Face Hub with transfer feature
        ),
        huggingface_hub(
            # temp_dir_root="/metaflow_temp/hf_hub"
            # If you use use_tmpfs=True, then you can uncomment this.
        ),
        environment(
            vars={
                "HF_HUB_ENABLE_HF_TRANSFER": "1",  # Enable Hugging Face transfer acceleration
            }
        ),
    ]
    for deco in deco_list:
        func = deco(func)
    return func


def training_environment(func):
    deco_list = [
        card(),
        gpu_profile(interval=10),
        pypi(
            python="3.11.10",
            packages={
                "trl @ git+https://github.com/huggingface/trl": "@69ad852e5654a77f1695eb4c608906fe0c7e8624",
                "latex2sympy2_extended": "0.9.3",
                "transformers": "4.48.1",
                "math-verify": "0.3.3",
                "wandb": "0.19.5",
                "deepspeed": "0.16.3",
                "vllm": "v0.7.0",
            },
        ),
        environment(
            vars={
                "WANDB_PROJECT": "grpo",
                "WANDB_LOG_MODEL": "false",
                "NCCL_IB_HCA": "mlx5",
                "UCX_NET_DEVICES": "mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1",
                "SHARP_COLL_ENABLE_PCI_RELAXED_ORDERING": "1",
                "NCCL_COLLNET_ENABLE": "0",
            }
        ),
    ]
    for deco in deco_list:
        func = deco(func)
    return func


def attach_decos(deco_list):
    def decorator(func):
        for deco in deco_list:
            func = deco(func)
        return func

    return decorator


class GRPOFlow(FlowSpec):

    accelerate_config = Config(
        "accelerate_config",
        default="accelerate_configs/zero3.yaml",
        parser="yaml.full_load",
    )

    training_config = IncludeFile(
        "config",
        default="config.yaml",
        is_text=True,
    )

    dry_run = Parameter("dry-run", default=False, type=bool)

    @step
    def start(self):
        self.next(self.pull_model)

    @huggingface
    @kubernetes(**H100_K8S_CONFIG)
    @step
    def pull_model(self):
        import yaml
        import time

        config = yaml.safe_load(self.training_config)
        self.model_name = config["model_name_or_path"]
        current.run.add_tag("model:%s" % self.model_name)

        start_time = time.time()
        self.llama_model = current.huggingface_hub.snapshot_download(
            repo_id=self.model_name,
            allow_patterns=[
                "*.safetensors",
                "*.json",
                "tokenizer.*",
            ],  # Download only model weights and tokenizer files
            max_workers=100,  # Use up to 100 threads for parallel download
        )
        end_time = time.time()
        self.time_taken = end_time - start_time
        self.next(
            self.train,
            num_parallel=config["num_nodes"],
        )

    @torchrun
    @model(
        load=["llama_model"],
        # temp_dir_root="/metaflow_temp/models"
        # If you use use_tmpfs=True, then you can uncomment this.
    )
    @training_environment
    @kubernetes(**H100_K8S_CONFIG)
    @step
    def train(self):
        """Run GRPO training with accelerate"""

        import yaml

        config = yaml.safe_load(self.training_config)
        if self.dry_run:
            config.update(DRY_RUN_PARAMS)
            print(
                "Using dry run params",
            )

        config["run_name"] = current.pathspec
        config["output_dir"] = "./output"
        config["model_name_or_path"] = current.model.loaded["llama_model"]

        acc_config = self.accelerate_config.to_dict()
        accelerate = Accelerate(acc_config, use_multi_node_config=True)

        if config["use_vllm"] and len(accelerate.multi_node_config) > 0:
            accelerate.multi_node_config["num_processes"] = (
                accelerate.multi_node_config["num_processes"] - 1
            )

        accelerate.launch(
            entry_point="grpo.py",
            config_dict=config,
        )

        if current.parallel.node_index == 0:
            current.model.save(
                "./output",
                storage_format="files",
            )

        self.next(self.join)

    @step
    def join(self, inputs):
        self.next(self.end)

    @step
    def end(self):
        """End of flow"""
        pass


if __name__ == "__main__":
    GRPOFlow()
