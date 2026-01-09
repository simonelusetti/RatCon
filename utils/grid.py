"""Run simple override sweeps defined in `grid.yaml`.

The YAML file must expose two top-level keys:

```
baseline:
  - data.train.subset=0.1
  - train.epochs=5

sweep:
  - - model.loss.l_s=0.01
    - model.loss.l_tv=5.0
  - - model.loss.l_s=0.05
    - model.loss.l_tv=2.0
```

Each list under `sweep` is appended to the baseline overrides and executed
once via `dora run`. The script prints a compact summary of
successes/failures and the run signatures for each sweep setting. You can
also call `--combine-plots` to build a single grid PNG from existing
`outputs/plots/<sig>/rates_cath.png` files using `.argv.json` as titles.
"""


import subprocess, yaml
from datetime import datetime
from pathlib import Path


import yaml
from tabulate import tabulate
from tqdm import tqdm


CONFIG_PATH = Path("./utils/grid.yaml")
RUNS_DIR = Path("../outputs/grids")


def load_config(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    baseline = data.get("baseline", []) or []
    sweep = data.get("sweep", []) or []
    if not sweep:
        raise ValueError("No sweep entries defined in config")
    return baseline, sweep


def run_and_capture_signature(cmd, pbar):
    signature = None
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        line = line.rstrip()
        if line:
            pbar.write(line)
        if "Exp signature:" in line:
            signature = line.split("Exp signature:")[-1].strip()
    if not signature:
        raise RuntimeError("Experiment signature not found in output")
    returncode = process.wait()
    if returncode != 0:
        raise subprocess.CalledProcessError(returncode, cmd)
    return signature


def main():
    try:
        baseline, sweep = load_config(CONFIG_PATH)
    except (FileNotFoundError, ValueError) as exc:
        print(f"⚠️ {exc}")
        return

    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    summary_rows = []

    total_runs = len(sweep)
    with tqdm(total=total_runs, desc="Grid sweep", unit="run") as pbar:
        for overrides in sweep:
            setting_overrides = list(baseline) + list(overrides)
            setting_label = " ".join(overrides) if overrides else "<no overrides>"

            pbar.write(f"\n=== Setting: {setting_label} ===")
            pbar.write(f"Baseline overrides: {baseline}")
            pbar.write(f"Sweep overrides   : {overrides}")

            cmd = ["dora", "run"] + setting_overrides
            signatures = []
            failures = 0
            try:
                sig = run_and_capture_signature(cmd, pbar)
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
