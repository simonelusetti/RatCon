# monkey_grid.py
import itertools
import subprocess

# Define the sweep values
modes = ["lowpass","highpass","bandpass"]
keep_ratio = ["0.3","0.5","0.7"]

baseline = ["data.train.subset=0.1","model.fourier.use=True"]

params_grid = itertools.product(keep_ratio, modes)
n = len(list(params_grid))
i = 0

for kr, mode in itertools.product(keep_ratio, modes):
    # Build the argument list
    overrides = [
        f"model.fourier.mode={mode}",
        f"model.fourier.keep_ratio={kr}",
    ]

    # Print what we’re about to run
    print(f"\n=== Running Exp {i+1}/{n} with {baseline} {overrides} ===")

    # Call `dora run` with these overrides
    cmd = ["dora", "run"] + baseline + overrides
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Run failed: {overrides} -> {e}")
    
    i+=1
