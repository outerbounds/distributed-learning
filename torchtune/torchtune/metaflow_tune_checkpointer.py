import glob
from typing import Any, Dict, Optional
import sys
from torchtune.training.checkpointing import FullModelHFCheckpointer
from torchtune.training.checkpointing._utils import SUFFIXES_TO_NOT_COPY
from metaflow import Checkpoint
from async_checkpoint import AsyncCheckpoint
import os
from torchtune.utils._logging import get_logger, log_rank_zero
import shutil
import torch
from torchtune import utils

logger = get_logger("DEBUG")


def recipe_from_checkpoint(chckpt_info, load_dir):
    Checkpoint().load(chckpt_info["metadata"]["recipe_checkpoint_key"], load_dir)


import os


def get_directory_tree(root_dir, indent=""):
    """Returns the directory tree structure as a string starting from the given root directory."""
    tree_str = ""
    for item in os.listdir(root_dir):
        item_path = os.path.join(root_dir, item)
        if os.path.isfile(item_path):
            tree_str += f"{indent}├── {item}\n"
        elif os.path.isdir(item_path):
            tree_str += f"{indent}├── {item}/\n"
            tree_str += get_directory_tree(item_path, indent + "│   ")
    return tree_str


class MetaflowFullModelCheckpointer(FullModelHFCheckpointer):
    def __init__(
        self,
        *args,
        use_async_checkpoint: bool = False,
        resume_from_checkpoint: bool = False,
        should_load_recipe_state: bool = False,
        max_checkpoints_on_disk: Optional[int] = None,
        delete_original_checkpoints_on_load: bool = False,
        ephemeral_recipe_checkpoints: bool = False,
        sync_recipe_checkpoints: bool = False,
        **kwargs,
    ):
        _resume_from_checkpoint = resume_from_checkpoint or should_load_recipe_state

        self.max_checkpoints_on_disk = max_checkpoints_on_disk
        if max_checkpoints_on_disk is not None:
            assert (
                max_checkpoints_on_disk >= 0
            ), "max_checkpoints_on_disk must be greater than 0"
        self._checkpoints_saved_to_disk = list()
        # We will custom set things here like the
        #   - recipe checkpoint path
        #   - adapter checkpoint path
        # Few reasons for doing so:
        # 1. Metaflow's checkpoint logic will ensure that the model gets loaded to a certain path.
        # We will need to replace those files here in the checkpointer.
        # 2. TT has this weird logic where it looks for certain files between output_dir and checkpoint_dir
        # when resuming training from a checkpoint.
        # 3. The simpler thing to do is that all checkpoints get loaded from the `checkpoint_dir` and all outputs
        # get written to the outputs dir. If users want to resume training runs, they explicitly set the `checkpoint_dir` to the
        # path of the model and separately set the output dir while explicitly passing the recipe path. This decouples all
        # configurations for different settings.
        super().__init__(
            *args,
            resume_from_checkpoint=False,
            should_load_recipe_state=False,
            **kwargs,
        )
        self.mf_checkpoint = None
        self.use_async_checkpoint = use_async_checkpoint
        if _resume_from_checkpoint:
            _recipe_checkpoint = kwargs.get("recipe_checkpoint", None)
            if _recipe_checkpoint:
                self._recipe_checkpoint = _recipe_checkpoint
            logger.info(
                "Resuming from checkpoint using:"
                f"\n\tcheckpoint_paths: {[str(path) for path in self._checkpoint_paths]}"
                f"\n\trecipe_checkpoint: {self._recipe_checkpoint}"
                f"\n\tadapter_checkpoint: {self._adapter_checkpoint}"
            )

        self._resume_from_checkpoint = resume_from_checkpoint
        self._should_load_recipe_state = should_load_recipe_state
        self._delete_original_checkpoints_on_load = delete_original_checkpoints_on_load
        self._ephemeral_recipe_checkpoints = ephemeral_recipe_checkpoints
        self._sync_recipe_checkpoints = sync_recipe_checkpoints

    def load_checkpoint(self):
        loaded_checkpoint = super().load_checkpoint()
        if self._delete_original_checkpoints_on_load:
            world_size, rank = utils.get_world_size_and_rank()
            # Set a barrier so that we can ensure that checkpoints have been loaded in memory and there is a
            # consistent view of the checkpoint paths across all processes.
            print(
                "[@checkpoint] Deleting original checkpoints on load\n\t",
                "\n\t".join([str(x) for x in self._checkpoint_paths]),
                file=sys.stderr,
            )
            if world_size > 1:
                print(
                    "[@checkpoint] Setting a barrier to ensure that all processes have loaded the checkpoints",
                    world_size,
                    rank,
                    file=sys.stderr,
                )
                torch.distributed.barrier()
            if rank == 0:
                # TODO : run an all-gather to figure out if the all the procesess within the same node have the same checkpoint paths
                for checkpoint_path in self._checkpoint_paths:
                    if os.path.exists(checkpoint_path):
                        os.remove(str(checkpoint_path))

                if self._resume_from_checkpoint and self._recipe_checkpoint is not None:
                    print(
                        "[@checkpoint] Deleting recipe checkpoint on load\n\t",
                        str(self._recipe_checkpoint),
                        file=sys.stderr,
                    )
                    os.remove(str(self._recipe_checkpoint))

        return loaded_checkpoint

    def _delete_stale_checkpoints(self):
        if self.max_checkpoints_on_disk is not None:
            if len(self._checkpoints_saved_to_disk) > self.max_checkpoints_on_disk:
                print(
                    f"[@checkpoint] Deleting stale checkpoints from disk to conserve space: {self._checkpoints_saved_to_disk[0]}",
                    file=sys.stderr,
                )
                while (
                    len(self._checkpoints_saved_to_disk) > self.max_checkpoints_on_disk
                ):
                    old_checkpoint = self._checkpoints_saved_to_disk.pop(0)
                    if os.path.exists(old_checkpoint):
                        print(
                            f"[@checkpoint] Deleting checkpoint from disk: {old_checkpoint}",
                            file=sys.stderr,
                        )
                        shutil.rmtree(old_checkpoint)
        else:
            print(
                "[@checkpoint] max_checkpoints_on_disk is not set, not deleting any checkpoints",
                file=sys.stderr,
            )

    def _clear_recipe_checkpoints_from_output_dir(self):
        if self._ephemeral_recipe_checkpoints:
            recipe_state = os.path.join(
                self._output_dir, "recipe_state", "recipe_state.pt"
            )
            if os.path.exists(recipe_state):
                print(
                    f"[@checkpoint] Deleting recipe checkpoint from output dir: {recipe_state}",
                    file=sys.stderr,
                )
                os.remove(recipe_state)

    def save_checkpoint(
        self,
        state_dict: Dict[str, Any],
        epoch: int,
        intermediate_checkpoint: bool = False,
        adapter_only: bool = False,
    ) -> None:
        if self.use_async_checkpoint and self.mf_checkpoint is not None:
            # We join over here because we need to ensure that the recipe state
            # is not overwritten when we call super().save_checkpoint
            self.mf_checkpoint.join()

        # By this point we have already ended up saving any old checkpoints.
        # Any recipe checkpoint will get overwritten so we don't need to delete it.
        self._delete_stale_checkpoints()
        # Always clear the recipe checkpoints from the output dir
        # since they will be overwritten by the new recipe state
        # This way we can ensure that we have enough space for the new recipe state
        self._clear_recipe_checkpoints_from_output_dir()

        super().save_checkpoint(
            state_dict, epoch, intermediate_checkpoint, adapter_only
        )
        if self.mf_checkpoint is None:
            if self.use_async_checkpoint:
                self.mf_checkpoint = AsyncCheckpoint(
                    max_parallel_saves=2 if self._sync_recipe_checkpoints else 1
                )
            else:
                self.mf_checkpoint = Checkpoint()

        recipe_checkpoint = None
        model_metadata = {
            "epoch": epoch,
            "saved_by": "torchtune",
        }
        model_path = os.path.join(self._output_dir, "epoch_" + str(epoch))
        if intermediate_checkpoint:
            # save recipe state
            if self._sync_recipe_checkpoints:
                recipe_state = os.path.join(
                    self._output_dir, "recipe_state", "recipe_state.pt"
                )
                recipe_checkpoint_or_key = self.mf_checkpoint.save(
                    recipe_state,
                    metadata=model_metadata,
                    name="recipe_state",
                    latest=False,
                )
                print(
                    f"[@checkpoint] Saved recipe state",
                    recipe_checkpoint_or_key,
                    file=sys.stderr,
                )
                recipe_checkpoint = recipe_checkpoint_or_key
                if not self.use_async_checkpoint:
                    recipe_checkpoint = recipe_checkpoint_or_key["key"]
                model_metadata["recipe_checkpoint_key"] = recipe_checkpoint

            if self.max_checkpoints_on_disk is not None:
                self._checkpoints_saved_to_disk.append(model_path)

            model_checkpoint_or_key = self.mf_checkpoint.save(
                model_path, metadata=model_metadata, name="model", latest=True
            )
            print(
                f"[@checkpoint] Saved Model state",
                model_checkpoint_or_key,
                file=sys.stderr,
            )
            print(
                f"[@checkpoint] Directory tree for model path: {model_path}",
                get_directory_tree(model_path),
                file=sys.stderr,
            )
        else:
            _save_method = (
                self.mf_checkpoint.sync_save
                if self.use_async_checkpoint
                else self.mf_checkpoint.save
            )
            print(
                "[@checkpoint] Saving final checkpoint Synchronously!", file=sys.stderr
            )
            model_checkpoint = _save_method(
                model_path, metadata=model_metadata, name="model", latest=True
            )
            if self.use_async_checkpoint:
                print(
                    "[@checkpoint] Waiting for async checkpoint to finish",
                    file=sys.stderr,
                )
                self.mf_checkpoint.join()

            print(
                f"[@checkpoint] Saved Final Model Checkpoint",
                model_checkpoint,
                file=sys.stderr,
            )
