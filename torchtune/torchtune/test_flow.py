from metaflow import FlowSpec, step, current, checkpoint
from async_checkpoint import AsyncCheckpoint
import time
import os
import tempfile


class AsyncCheckpointTestFlow(FlowSpec):
    @checkpoint
    @step
    def start(self):
        # Initialize async checkpoint with max 2 parallel saves
        async_ckpt = AsyncCheckpoint(max_parallel_saves=2)

        # Create some dummy data to save
        with tempfile.TemporaryDirectory() as temp_dir:

            # Save checkpoints in a loop
            for i in range(10):
                # Update the file with new data
                with open(os.path.join(temp_dir, f"checkpoint_{i}.txt"), "w") as f:
                    f.write(f"data from iteration {i}")

                # Save checkpoint asynchronously
                checkpoint_key = async_ckpt.save(
                    path=os.path.join(temp_dir, f"checkpoint_{i}.txt"),
                    metadata={"iteration": i},
                    name=f"checkpoint_{i}",
                )
                print(f"Initiated save for iteration {i}: {checkpoint_key}")

                # Simulate some work
                time.sleep(1)

            print("Waiting for all checkpoints to complete...")
            async_ckpt.join()
        # Wait for all checkpoints to complete

        # Get metadata from all checkpoints
        self.checkpoint_metadata = async_ckpt.get_metadata()

        self.next(self.end)

    @step
    def end(self):
        print("\nCheckpoint Metadata:")
        for metadata in self.checkpoint_metadata:
            print(f"Metadata: {metadata}")


if __name__ == "__main__":
    AsyncCheckpointTestFlow()
