import os
import re
import subprocess, yaml
from datetime import datetime
from pathlib import Path


from tabulate import tabulate
from tqdm import tqdm

_ANSI_ESCAPE = re.compile(r'\x1b\[[0-9;]*[A-Za-z]|\r')


CONFIG_PATH = Path("./utils/grid.yaml")
RUNS_DIR = Path("outputs/utils/grid")
DEFAULT_CFG_PATH = Path("./src/conf/default.yaml")


def load_config(path: Path) -> tuple[list[str], list[list[str]]]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    baseline = data.get("baseline", []) or []
    sweep = data.get("sweep", []) or []
    if not sweep:
        raise ValueError("No sweep entries defined in config")
    return baseline, sweep


def load_default_train_epochs(path: Path) -> int:
    if not path.exists():
        return 30

    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}

    train_cfg = data.get("train", {}) or {}
    value = train_cfg.get("epochs")
    try:
        return int(value)
    except (TypeError, ValueError):
        return 30


def extract_override_value(overrides: list[str], key: str) -> str | None:
    prefix = f"{key}="
    for item in reversed(overrides):
        if item.startswith(prefix):
            return item[len(prefix):]
    return None


def resolve_train_epochs(baseline: list[str], overrides: list[str]) -> int:
    value = extract_override_value(list(baseline) + list(overrides), "train.epochs")
    if value is None:
        return load_default_train_epochs(DEFAULT_CFG_PATH)
    try:
        return int(value)
    except ValueError:
        return load_default_train_epochs(DEFAULT_CFG_PATH)


def run_and_capture_signature(cmd: list[str], pbar: tqdm, run_pbar: tqdm | None = None) -> str:
    signature = None
    child_env = {
        **os.environ,
        "DISABLE_TQDM": "1",
        "TQDM_DISABLE": "1",
        "PYTHONUNBUFFERED": "1",
        "HF_DATASETS_DISABLE_PROGRESS_BARS": "1",
        "DATASETS_DISABLE_PROGRESS_BARS": "1",
    }
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=child_env,
    )
    assert process.stdout is not None
    for line in process.stdout:
        line = _ANSI_ESCAPE.sub('', line).rstrip()
        if run_pbar is not None and "GRID_EPOCH " in line:
            step = line.rsplit("GRID_EPOCH ", 1)[1]
            try:
                current, total = step.split("/", 1)
                current_i = int(current)
                total_i = int(total)
                if run_pbar.total != total_i:
                    run_pbar.total = total_i
                if current_i > run_pbar.n:
                    run_pbar.update(current_i - run_pbar.n)
            except ValueError:
                pbar.write(line)
            continue
        if line:
            pbar.write(line)
        if "Exp signature:" in line:
            signature = line.split("Exp signature:")[-1].strip()
    if not signature:
        raise RuntimeError("Experiment signature not found in output")
    returncode = process.wait()
    if returncode == 0 and run_pbar is not None and run_pbar.total is not None and run_pbar.n < run_pbar.total:
        run_pbar.update(run_pbar.total - run_pbar.n)
    if returncode != 0:
        raise subprocess.CalledProcessError(returncode, cmd)
    return signature

def shorten_key(key: str, keep_parts: int = 1) -> str:
    parts = key.split(".")
    return ".".join(parts[-keep_parts:])

def main() -> None:
    try:
        baseline, sweep = load_config(CONFIG_PATH)
    except (FileNotFoundError, ValueError) as exc:
        print(f"⚠️ {exc}")
        return

    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    summary_rows = []

    total_runs = len(sweep)
    with tqdm(total=total_runs, desc="Grid sweep", unit="run", leave=True, dynamic_ncols=True) as pbar:
        for overrides in sweep:
            setting_overrides = list(baseline) + list(overrides)
            setting_label = " ".join(overrides) if overrides else "<no overrides>"
            train_epochs = resolve_train_epochs(baseline, overrides)

            pbar.write(f"=== Setting: {setting_label} ===")
            pbar.write(f"Baseline overrides: {baseline}")
            pbar.write(f"Sweep overrides   : {overrides}")

            cmd = ["dora", "run"] + setting_overrides
            signatures = []
            failures = 0
            run_desc = f"Training {shorten_key(setting_label)}"
            with tqdm(
                total=train_epochs,
                desc=run_desc,
                unit="epoch",
                leave=False,
                dynamic_ncols=True,
            ) as run_pbar:
                try:
                    sig = run_and_capture_signature(cmd, pbar, run_pbar=run_pbar)
                    signatures.append(sig)
                    pbar.write(f"    ✓ Run signature: {sig}")
                except subprocess.CalledProcessError as exc:
                    failures += 1
                    pbar.write(f"    ⚠️ Run failed: {exc}")
                except RuntimeError as exc:
                    failures += 1
                    pbar.write(f"    ⚠️ {exc}")
                finally:
                    pbar.update(1)

            summary_rows.append(
                {
                    "setting": setting_label,
                    "runs": len(signatures),
                    "failures": failures,
                    "signatures": ", ".join(signatures) if signatures else "",
                }
            )

    if not summary_rows:
        print("No runs executed.")
        return

    table = tabulate(summary_rows, headers="keys", tablefmt="github", floatfmt=".4f")
    print("\nSummary table:\n")
    print(table)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_path = RUNS_DIR / f"grid_results_{timestamp}.md"
    output_content = f"Baseline overrides: {', '.join(baseline) if baseline else '<none>'}\n\n{table}\n"
    output_path.write_text(output_content, encoding="utf-8")
    print(f"\nSaved table view to {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Grid sweep runner and plot combiner.")
    args = parser.parse_args()

    main()
