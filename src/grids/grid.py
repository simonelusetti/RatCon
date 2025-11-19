"""Dora grid definition that mirrors a YAML-driven sweep."""

from pathlib import Path
import yaml
import itertools
from ._history_utils import RatConExplorer

CONFIG_PATH = "reference_sentence.yaml"

def load_yaml_sweep(path: Path):
    """Load baseline and sweep configurations from a YAML file."""
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    baseline = data.get("baseline", {})
    sweep = data.get("sweep", {})

    if not isinstance(sweep, dict):
        raise ValueError("'sweep' must be a dictionary of lists")

    keys = list(sweep.keys())
    values = list(sweep.values())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    return baseline, combinations


@RatConExplorer
def explorer(launcher):
    """Run experiments defined by a YAML grid.

    Args:
        launcher: Dora launcher
        config_path (str | Path, optional): Path to the YAML config.
            Defaults to 'grid.yaml' in the same directory as this file.
    """
    # Default to "grid.yaml" next to this file
    config_path = Path(__file__).resolve().parent / CONFIG_PATH

    baseline, combinations = load_yaml_sweep(config_path)
    configured_launcher = launcher.bind(baseline) if baseline else launcher

    print(baseline, combinations)

    for overrides in combinations:
        configured_launcher(overrides)
