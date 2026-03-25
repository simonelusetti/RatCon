from __future__ import annotations

# -----------------------------------------------------------------------------
# Configure here
# -----------------------------------------------------------------------------

EXPERIMENTS: list[str] = [
]

# -----------------------------------------------------------------------------

import argparse
import subprocess
import sys
from pathlib import Path

from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
XPS_DIR = PROJECT_ROOT / "outputs" / "xps"
CHI_PLOT = Path("plots/chi_square.png")
EXP_STEPS = 1


def has_checkpoint(sig_dir: Path) -> bool:
    return any(sig_dir.glob("state/models/model_*.pth"))


def available_signatures() -> list[str]:
    if not XPS_DIR.exists():
        return []
    return sorted([p.name for p in XPS_DIR.iterdir() if p.is_dir()])


def resolve_targets(from_top_list: list[str], use_all: bool) -> list[str]:
    all_sigs = available_signatures()

    # Normalize list values in case users leave placeholders like "".
    configured = [sig.strip() for sig in from_top_list if sig and sig.strip()]

    # --all always wins, even when EXPERIMENTS is non-empty.
    if use_all:
        return all_sigs

    # Empty (or effectively empty) EXPERIMENTS behaves like --all.
    if not configured:
        return all_sigs

    return configured


def run_eval(sig: str, dry_run: bool = False, current_bar: tqdm | None = None) -> tuple[bool, str]:
    sig_dir = XPS_DIR / sig
    if not sig_dir.exists():
        return False, f"Missing signature folder: {sig}"

    cmd = [
        "dora",
        "run",
        "--from_sig",
        sig,
        "train.no_train=true",
        "runtime.grid=false",
        "runtime.eval.skip_stsb=true",
        "runtime.eval.sweep_range=[0.1, 1.0, 10]",
    ]

    if dry_run:
        return True, " ".join(cmd)

    proc = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        text=True,
        check=False,
    )

    if proc.returncode != 0:
        return False, f"dora failed for {sig} (exit={proc.returncode})"

    plot_path = sig_dir / CHI_PLOT
    if not plot_path.exists():
        return False, f"Eval finished but no chi-square plot found for {sig}: {plot_path}"

    return True, str(plot_path)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Batch final eval for Dora signatures (equivalent to train.no_train=true)."
        )
    )
    ap.add_argument(
        "--all",
        action="store_true",
        help="Evaluate all signatures found in outputs/xps.",
    )
    ap.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip signatures that already have plots/chi_square.png.",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them.",
    )
    ap.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on first failure.",
    )
    args = ap.parse_args()

    targets = resolve_targets(EXPERIMENTS, args.all)
    if not targets:
        print("No signatures selected.")
        return

    ok = 0
    failed = 0
    skipped = 0
    missing_ckpt = 0

    tqdm.write(f"Selected {len(targets)} signatures.")

    overall_bar = tqdm(
        total=len(targets),
        desc="Overall",
        position=0,
        leave=True,
        dynamic_ncols=True,
    )
    current_bar = tqdm(
        total=EXP_STEPS,
        desc="Current: idle",
        position=1,
        leave=False,
        dynamic_ncols=True,
    )

    try:
        for i, sig in enumerate(targets, start=1):
            sig_dir = XPS_DIR / sig
            plot_path = sig_dir / CHI_PLOT

            current_bar.reset(total=EXP_STEPS)
            current_bar.set_description_str(f"Current: {sig} | queued")
            current_bar.set_postfix_str("")

            if args.skip_existing and plot_path.exists():
                skipped += 1
                current_bar.set_description_str(f"Current: {sig} | skipped")
                if current_bar.n < current_bar.total:
                    current_bar.update(current_bar.total - current_bar.n)
                tqdm.write(f"[{i}/{len(targets)}] SKIP  {sig} (already has {CHI_PLOT})")
                overall_bar.update(1)
                continue

            if not has_checkpoint(sig_dir):
                missing_ckpt += 1
                current_bar.set_description_str(f"Current: {sig} | no-ckpt")
                if current_bar.n < current_bar.total:
                    current_bar.update(current_bar.total - current_bar.n)
                tqdm.write(f"[{i}/{len(targets)}] SKIP  {sig} (no checkpoint in state/models)")
                overall_bar.update(1)
                continue

            current_bar.set_description_str(f"Current: {sig} | run")
            success, message = run_eval(sig, dry_run=args.dry_run, current_bar=current_bar)

            if current_bar.n < current_bar.total:
                current_bar.update(current_bar.total - current_bar.n)

            if success:
                ok += 1
                tag = "DRY" if args.dry_run else "OK"
                current_bar.set_description_str(f"Current: {sig} | done")
                tqdm.write(f"[{i}/{len(targets)}] {tag}   {sig} -> {message}")
            else:
                failed += 1
                current_bar.set_description_str(f"Current: {sig} | fail")
                tqdm.write(f"[{i}/{len(targets)}] FAIL  {sig}\n{message}\n")
                overall_bar.update(1)
                if args.fail_fast:
                    break

            overall_bar.update(1)
    finally:
        current_bar.close()
        overall_bar.close()

    print("\nSummary")
    print(f"  total:   {len(targets)}")
    print(f"  ok:      {ok}")
    print(f"  failed:  {failed}")
    print(f"  skipped: {skipped}")
    print(f"  no-ckpt: {missing_ckpt}")

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
