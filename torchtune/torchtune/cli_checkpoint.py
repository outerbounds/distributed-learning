from metaflow._vendor import click
from metaflow import JSONType
from metaflow import Checkpoint
import json


@click.command()
@click.option("--checkpoint-kwargs", type=JSONType, help="Checkpoint kwargs to save")
@click.option(
    "--output-metadata-path",
    type=click.Path(exists=True),
    help="Path to save checkpoint metadata",
)
@click.option("--checkpoint-version", type=int, help="Checkpoint version to save")
def save(
    checkpoint_kwargs,
    output_metadata_path=None,
    checkpoint_version=None,
):
    checkpoint = Checkpoint()
    checkpoint = checkpoint._init_checkpoint_for_writes(checkpoint)
    if checkpoint_version is not None:
        checkpoint._checkpointer._set_current_version(checkpoint_version)

    checkpoint_ref_dict = checkpoint.save(**checkpoint_kwargs)
    if output_metadata_path:
        with open(output_metadata_path, "w") as f:
            json.dump(checkpoint_ref_dict, f)


if __name__ == "__main__":
    save()
