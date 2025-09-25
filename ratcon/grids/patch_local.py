# ratcon/grids/patch_local.py
import os
import pickle
import signal
import subprocess as sp
from pathlib import Path

import submitit
from dora import shep


def _get_submitit_executor_local(self, name: str, folder: Path, slurm_config):
    """
    Replacement for Shepherd._get_submitit_executor:
    Uses LocalExecutor instead of SlurmExecutor.
    """
    folder.mkdir(parents=True, exist_ok=True)
    # Local executor does not use slurm_config at all
    return submitit.LocalExecutor(folder=folder)


def _local_state(job):
    """
    Map LocalExecutor jobs to simple states: PENDING, RUN, ERR, COM.
    """
    if job is None:
        return None
    if job.done():
        try:
            result = job.result()  # raises if failed
            return "COMPLETED"
        except Exception:
            return "FAILED"
    else:
        return "RUNNING"


def sheep_state(self, mode="standard"):
    if self.job is None:
        return None
    return _local_state(self.job)


def sheep_is_done(self, mode="standard"):
    if self.job is None:
        return True
    return self.job.done()


# Apply patches
shep.Shepherd._get_submitit_executor = _get_submitit_executor_local
shep.Sheep.state = sheep_state
shep.Sheep.is_done = sheep_is_done
