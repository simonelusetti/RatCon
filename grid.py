# monkey_grid.py
import itertools
import subprocess

# Define the sweep values
l_comp_values = [0.1, 0.5, 1.0]
l_s_values = [0.1, 0.5, 1.0]
l_tv_values = [0.1, 0.5, 1.0]

baseline = ["data.train.subset=0.1"]

for l_comp, l_s, l_tv in itertools.product(l_comp_values, l_s_values, l_tv_values):
    # Build the argument list
    overrides = [
        f"model.loss.l_comp={l_comp}",
        f"model.loss.l_s={l_s}",
        f"model.loss.l_tv={l_tv}",
    ]

    # Print what we’re about to run
    print(f"\n=== Running {baseline} {overrides} ===")

    # Call `dora run` with these overrides
    cmd = ["dora", "run"] + baseline + overrides
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Run failed: {overrides} -> {e}")
