from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import (
    plot_chi_square_overview,
    plot_loss_overview,
    plot_selection_rates_overview,
    plot_spearman_overview,
)
from utils.dora_utils import (
    XPS_DIR,
    build_groups,
    expected_checkpoint,
    filter_sig_dirs_by_group_size,
    load_dora_exclude,
    load_run,
    needs_eval,
    rerun_eval,
)


OUT_ROOT = PROJECT_ROOT / "outputs" / "utils" / "overview"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render grouped overview figures (loss, chi-square, spearman) as mean+-std across runs."
    )
    parser.add_argument("--sigs", nargs="*", default=None, help="Optional list of signatures to include.")
    parser.add_argument("--rerun-eval", action="store_true", help="Force re-run eval on each selected signature.")
    parser.add_argument("--dry-run-eval", action="store_true", help="Print eval commands only.")
    parser.add_argument("--skip-auto-eval", action="store_true", help="Skip automatic eval for missing data.")
    parser.add_argument("--ncols", type=int, default=4, help="Grid columns for overview figures.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Custom output directory.")
    parser.add_argument(
        "--min-group-runs",
        type=int,
        default=1,
        help="Keep only groups with at least this many runs (after Dora excludes and run key filtering).",
    )
    parser.add_argument(
        "--only-multi-run-groups",
        action="store_true",
        help="Convenience flag for --min-group-runs=2.",
    )
    args = parser.parse_args()

    if args.dry_run_eval and not args.rerun_eval:
        raise ValueError("--dry-run-eval requires --rerun-eval")
    if args.min_group_runs < 1:
        raise ValueError("--min-group-runs must be >= 1")

    min_group_runs = max(args.min_group_runs, 2) if args.only_multi_run_groups else args.min_group_runs

    if args.sigs:
        sig_dirs = [XPS_DIR / sig for sig in args.sigs]
    else:
        sig_dirs = sorted([p for p in XPS_DIR.iterdir() if p.is_dir()], key=lambda p: p.name)

    for sig_dir in sig_dirs:
        if not sig_dir.exists():
            raise FileNotFoundError(f"Missing signature directory: {sig_dir}")

    exclude_patterns = load_dora_exclude()
    sig_dirs = filter_sig_dirs_by_group_size(sig_dirs, exclude_patterns, min_group_runs)
    if not sig_dirs:
        raise ValueError("No signatures left after group-size filtering.")

    if not args.skip_auto_eval:
        missing_eval = [sd for sd in sig_dirs if needs_eval(sd)]
        if missing_eval:
            print(f"Found {len(missing_eval)} runs with missing evaluation data.")
            print("Auto-triggering evaluation...")
            for sig_dir in missing_eval:
                rerun_eval(sig_dir.name, sig_dir, expected_checkpoint(sig_dir), dry_run=args.dry_run_eval)
            print()

    if args.rerun_eval:
        print(f"Force re-running evaluation for all {len(sig_dirs)} selected signatures...")
        for sig_dir in sig_dirs:
            rerun_eval(sig_dir.name, sig_dir, expected_checkpoint(sig_dir), dry_run=args.dry_run_eval)
        print()

    runs = []
    for sig_dir in sig_dirs:
        try:
            runs.append(load_run(sig_dir))
        except FileNotFoundError as exc:
            print(f"Skipping {sig_dir.name}: {exc}")

    if not runs:
        raise ValueError("No valid runs found after loading signatures.")

    groups = build_groups(runs, exclude_patterns)
    if not groups:
        raise ValueError("No experiment groups found.")

    out_root = args.output_dir or OUT_ROOT
    out_root.mkdir(parents=True, exist_ok=True)

    for child in out_root.iterdir():
        if child.is_dir():
            shutil.rmtree(child)

    plot_loss_overview(groups, out_root / "loss_overview.png", ncols=args.ncols)
    plot_selection_rates_overview(groups, out_root / "selection_rates_overview.png", ncols=args.ncols)
    plot_chi_square_overview(groups, out_root / "chi_square_overview.png", ncols=args.ncols, metric="chi_square")
    plot_chi_square_overview(groups, out_root / "cramers_v_overview.png", ncols=args.ncols, metric="cramers_v")
    plot_spearman_overview(groups, out_root / "spearman_overview.png", ncols=args.ncols)

    print(f"Saved overview figures to {out_root}")


if __name__ == "__main__":
    main()
