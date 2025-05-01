# Modes of async checkpointing:
# 1. Wait for a

import os
from multiprocessing import Queue
from subprocess import Popen, DEVNULL, PIPE
import json
import sys
import tempfile
import time
import uuid
from metaflow import Checkpoint

CLI_PATH = os.path.join(
    os.path.dirname(__file__), "cli_checkpoint.py"
)  # todo change this when the cli is moved

# TODO: Add methods to debug this thing.


class ProcessStatus:
    NOT_STARTED = "not started"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class ProcessManager:
    CurrentProcesses = {
        # proc_id: {
        #     "process": CheckpointProcess,
        #     "checkpoint_kwargs": checkpoint_kwargs,
        #     "version": version,
        #     "created_at": time.time(),
        # }
    }

    @classmethod
    def get(cls, proc_id):
        return cls.CurrentProcesses[proc_id]["process"]

    @classmethod
    def add(cls, proc_id, process, checkpoint_kwargs, version):
        cls.CurrentProcesses[proc_id] = {
            "process": process,
            "checkpoint_kwargs": checkpoint_kwargs,
            "version": version,
            "created_at": time.time(),
        }

    @classmethod
    def process_status(cls, proc_id):
        return cls.CurrentProcesses[proc_id]["process"].status()

    @classmethod
    def remove(cls, proc_id):
        del cls.CurrentProcesses[proc_id]

    @classmethod
    def kill(cls, proc_id):
        cls.CurrentProcesses[proc_id]["process"].kill()
        cls.remove(proc_id)

    @classmethod
    def status(cls):
        statuses = {}
        for proc_id in cls.CurrentProcesses:
            statuses[proc_id] = cls.CurrentProcesses[proc_id]["process"].status()
        return statuses

    @classmethod
    def num_active(cls):
        # number of processes that are running or haven't completed yet
        # for proc_id in cls.CurrentProcesses:
        #     process = cls.CurrentProcesses[proc_id]["process"]
        #     print(process.status(), proc_id,file=sys.stderr)
        #     if process.status() == ProcessStatus.FAILED:
        #         print(process.stderr(), file=sys.stderr)
        return sum(
            1
            for proc_id in cls.CurrentProcesses
            if cls.CurrentProcesses[proc_id]["process"].status()
            in [ProcessStatus.RUNNING, ProcessStatus.NOT_STARTED]
        )

    @classmethod
    def clear_stale_processes(cls):
        proc_ids_to_remove = []
        for proc_id in list(cls.CurrentProcesses):
            if cls.CurrentProcesses[proc_id]["process"].status() in [
                ProcessStatus.COMPLETED,
                ProcessStatus.FAILED,
            ]:
                proc_ids_to_remove.append(proc_id)

        for proc_id in proc_ids_to_remove:
            cls.finish_process(proc_id)

        metadata = []
        for proc_id in proc_ids_to_remove:
            metadata.append(cls.CurrentProcesses[proc_id]["process"].get_metadata())
            cls.remove(proc_id)

        return metadata

    @classmethod
    def finish_process(cls, proc_id):
        cls.CurrentProcesses[proc_id]["process"].finalize()
        # cls.remove(proc_id)

    @classmethod
    def get_metadata(cls):
        metadata = {}
        for proc_id in cls.CurrentProcesses:
            metadata[proc_id] = cls.CurrentProcesses[proc_id]["process"].get_metadata()
        return metadata


class CheckpointProcess:
    def __init__(self, checkpoint_kwargs, version):
        self.checkpoint_kwargs = checkpoint_kwargs
        self.version = version
        self.process = None
        self.temp_file = None
        self.metadata = None

    def stderr(self):
        if self.process is None:
            return None
        return self.process.stderr.read().decode("utf-8")

    def start(self):
        self.temp_file = tempfile.NamedTemporaryFile(dir=".")
        cmd = [
            sys.executable,
            CLI_PATH,
            "--checkpoint-kwargs",
            json.dumps(self.checkpoint_kwargs),
            "--output-metadata-path",
            self.temp_file.name,
            "--checkpoint-version",
            str(self.version),
        ]
        env = os.environ.copy()
        self.process = Popen(
            cmd,
            env=env,
            stderr=PIPE,
            stdout=PIPE,
        )

        return self

    def status(self):
        if self.process is None:
            return ProcessStatus.NOT_STARTED
        return_code = self.process.poll()
        if return_code is None:
            return ProcessStatus.RUNNING
        elif return_code == 0:
            return ProcessStatus.COMPLETED
        else:
            return ProcessStatus.FAILED

    def kill(self):
        if self.process is not None:
            if self.status() == ProcessStatus.RUNNING:
                self.process.kill()
            self.process = None
        if self.temp_file is not None:
            self.temp_file.close()
            self.temp_file = None

    def finalize(self):
        if self.status() == ProcessStatus.RUNNING:
            print(
                "[@checkpoint] Trying to finalize a running process, killing it",
                file=sys.stderr,
            )
        elif self.status() == ProcessStatus.COMPLETED:
            self.metadata = self._get_metadata()
        elif self.status() == ProcessStatus.FAILED:
            print(
                "[@checkpoint] Trying to finalize a failed checkpoint process.",
                file=sys.stderr,
            )
        self.kill()

    def get_metadata(self):
        return self.metadata

    def _get_metadata(self):
        with open(self.temp_file.name, "r") as f:
            return json.load(f)

    def wait(self):
        self.process.wait()

    def __del__(self):
        self.kill()


class AsyncCheckpoint:
    """

    This class provides asynchronous checkpointing capabilities by spawning a separate process to save checkpoints.
    This allows the main training process to continue without being blocked by checkpoint saves.
    If users need to load/list checkpoints they can use the usual APIs.
    Since this is experimental, we can integrate this API into the main api sometime in the future.

    Parameters
    ----------
    max_queue_length : int, default: 3
        Maximum number of checkpoints that can be queued for saving. If the queue is full,
        the save operation will block until space is available.

    Example
    -------
    ```python
    # Initialize async checkpoint
    async_ckpt = AsyncCheckpoint(max_queue_length=3)

    # Training loop
    for epoch in range(num_epochs):
        train_epoch()

        # Save checkpoint asynchronously
        async_ckpt.save(
            path="model.pt",
            metadata={"epoch": epoch},
            name=f"epoch_{epoch}"
        )

    # Wait for all checkpoints to complete before exiting
    async_ckpt.join()
    ```
    """

    def __init__(
        self,
        max_parallel_saves: int = 3,
    ):
        self.max_parallel_saves = max_parallel_saves
        self._current_queue = []
        self._process_manager = ProcessManager
        self._versions = {}
        self._checkpoint = Checkpoint()
        self._metadata = []

    def wait_for_all_processes_to_finish(self):
        while self._process_manager.num_active() > 0:
            for pid, status in self._process_manager.status().items():
                if self._process_manager.process_status(pid) == ProcessStatus.RUNNING:
                    print(
                        f"[@checkpoint] Waiting for process {pid} to finish",
                        file=sys.stderr,
                    )
                    self._process_manager.get(pid).wait()

    def block_until_finished(
        self,
    ):
        # todo : make this function write to stderr if its taking too long to finish
        log_every_n_seconds = 10
        last_log_time = time.time()
        condition = (
            lambda: self._process_manager.num_active() >= self.max_parallel_saves
        )
        while condition():
            time.sleep(0.1)
            if time.time() - last_log_time > log_every_n_seconds:
                print(
                    f"[@checkpoint] Waiting for {self._process_manager.num_active()} checkpoints to finish",
                    file=sys.stderr,
                )
                last_log_time = time.time()

    def _get_named_based_version(self, name):
        if name not in self._versions:
            self._versions[name] = 0
        else:
            self._versions[name] += 1
        return self._versions[name]

    def save(
        self,
        path=None,
        metadata=None,
        latest=True,
        name="mfchckpt",
        storage_format="files",
    ):
        # TODO : When files are yanked from under the hood, this thing fails in mysterious ways.
        # We need to find a fix for this!
        if self._process_manager.num_active() >= self.max_parallel_saves:
            self.block_until_finished()
            self._metadata.extend(self._process_manager.clear_stale_processes())

        proc_id = uuid.uuid4()
        version = self._get_named_based_version(name)
        checkpoint_key = self._checkpoint.generate_key(name, version)
        # todo: make path an absolute path
        checkpoint_kwargs = {
            "path": path,
            "metadata": metadata,
            "latest": latest,
            "name": name,
            "storage_format": storage_format,
        }
        self._process_manager.add(
            proc_id,
            CheckpointProcess(
                checkpoint_kwargs,
                version,
            ).start(),
            checkpoint_kwargs,
            version,
        )

        return checkpoint_key

    def join(self):
        print("[@checkpoint] Joining async checkpoint", file=sys.stderr)
        self.wait_for_all_processes_to_finish()
        print(
            "[@checkpoint] Currently active processes",
            self._process_manager.num_active(),
            file=sys.stderr,
        )
        self._metadata.extend(self._process_manager.clear_stale_processes())
        print("[@checkpoint] Joined async checkpoint", file=sys.stderr)

    def get_metadata(self):
        return self._metadata

    def sync_save(
        self,
        path=None,
        metadata=None,
        latest=True,
        name="mfchckpt",
        storage_format="files",
    ):
        _chckpt = Checkpoint()
        _chckpt = _chckpt._init_checkpoint_for_writes(_chckpt)
        _chckpt._checkpointer._set_current_version(self._get_named_based_version(name))
        return _chckpt.save(path, metadata, latest, name, storage_format)
