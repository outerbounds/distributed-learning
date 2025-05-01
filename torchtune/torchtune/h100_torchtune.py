from metaflow import (
    FlowSpec,
    step,
    current,
    Parameter,
    checkpoint,
    kubernetes,
    pypi,
    card,
    gpu_profile,
    model,
    environment,
    IncludeFile,
    torchrun,
    huggingface_hub,
    retry,
)
import os
from metaflow_utils import TorchTune, Accelerate

# Change this what ever you want.
k8s_config = dict(
    cpu=100,
    memory=900 * 1000,
    gpu=8,
    shared_memory=200 * 1000,
    # You can change this to any image that has nccl, and cuda installed.
    image="registry.hub.docker.com/valayob/nebius-nccl-pytorch:0.0.2",
    disk=1000 * 1000,
    # use_tmpfs=True, # Uncomment this if you want to load models to in-memory tempfs disk.
    security_context=dict(
        privileged=True,
    ),
)


def huggingface(func):
    deco_list = [
        huggingface_hub(),
        pypi(
            python="3.11.5",
            packages={
                "huggingface-hub[hf_transfer]": "0.25.2"
            },  # Installing Hugging Face Hub with transfer feature
        ),
        # TODO: add secrets here if required.
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
        gpu_profile(),
        pypi(
            python="3.11.10",
            packages={
                "torchtune": "0.6.1",
                "torch": "2.6.0",
                "torchao": "0.8.0",
                "wandb": "0.19.5",
                "kagglehub": "0.3.6",  # needed by torchtune.
                "datasets": "3.2.0",
            },
        ),
    ]
    for deco in deco_list:
        func = deco(func)
    return func


class TorchTuneFlow(FlowSpec):

    training_config = IncludeFile(
        "config",
        default="multi_node_configs/1b_config.yaml",
        is_text=True,
    )

    dry_run = Parameter("dry-run", default=False, type=bool)

    @step
    def start(self):
        self.next(self.pull_model)

    @huggingface
    @kubernetes
    @step
    def pull_model(self):
        # large model reference can be found here : Task("LargeModelUpload/6603/pull_model_from_huggingface/43278").data.very_large_model
        import yaml
        import time

        config = yaml.safe_load(self.training_config)
        self.model_name = config["huggingface"]["repo_id"]
        current.run.add_tag("model:%s" % self.model_name)

        start_time = time.time()
        self.llama_model = current.huggingface_hub.snapshot_download(
            repo_id=self.model_name,
            # force_download=True,
            allow_patterns=config["huggingface"]["allow_patterns"],
            # Download only model weights and tokenizer files
            max_workers=100,
            repo_type="model",
        )
        end_time = time.time()
        self.time_taken = end_time - start_time
        self.next(
            self.train,
            # [CHANGE-FOR-SINGLE-NODE] Comment this if you want to run on a single node.
            num_parallel=2,
        )

    def _load_full_model_from_checkpoint(self, config):
        if current.checkpoint.is_loaded:
            # If we have a checkpoint loaded because of some failure then
            # we will also load the recipe checkpoint if it exists.
            config["base_model_dir"] = current.checkpoint.directory
            if "recipe_checkpoint_key" in current.checkpoint.info.metadata:
                config["recipe_checkpoint_key"] = current.checkpoint.info.metadata[
                    "recipe_checkpoint_key"
                ]
                recipe_checkpoint_path = current.model.load(
                    config["recipe_checkpoint_key"]
                )
                config["checkpointer"]["recipe_checkpoint"] = os.path.join(
                    recipe_checkpoint_path, "recipe_state.pt"
                )
                config["resume_from_checkpoint"] = True
                print(
                    "Resuming from checkpoint recipe of task:",
                    current.checkpoint.info.pathspec,
                    recipe_checkpoint_path,
                )
        return config

    @retry(times=3)
    # [CHANGE-FOR-SINGLE-NODE] Comment `@torchrun` if you want to run on a single node.
    @torchrun
    @checkpoint
    @model(load=["llama_model"])
    @training_environment
    @kubernetes(**k8s_config)
    @step
    def train(self):
        """Run GRPO training with accelerate"""

        import yaml

        config = yaml.safe_load(self.training_config)

        # Failure based checkpoint reloading logic comes here!
        config = self._load_full_model_from_checkpoint(config)

        config["run_name"] = current.pathspec
        config["output_dir"] = "./output"
        config["base_model_dir"] = current.model.loaded["llama_model"]

        tune = TorchTune(
            use_multi_node_config=True,
            # [CHANGE-FOR-SINGLE-NODE]
            # If you want to run on a single node, then set this to False.
            # use_multi_node_config=False
        )

        tune.run(
            # Replace this with your own recipe.
            "distributed_ft_recipe.py",
            config_dict=config,
            additional_cli_options=[
                "--max-restarts",
                "4"  # Ensures the cluster can be restarted multiple times.
                # [CHANGE-FOR-SINGLE-NODE]
                # "--nproc-per-node", "8" # Uncomment this if you remove @torchrun and run on a single node.
            ],
        )

        self.model_ref = current.model.save(
            "./output",
            storage_format="files",
        )

        self.next(self.join)

    @step
    def join(self, inputs):
        """Join the training job"""
        self.next(self.end)

    @step
    def end(self):
        """End of flow"""
        pass


if __name__ == "__main__":
    TorchTuneFlow()
