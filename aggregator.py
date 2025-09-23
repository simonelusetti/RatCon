import os
import json
import pandas as pd
from tabulate import tabulate

def aggregate_histories(xp_root="outputs/xps"):
    results = []

    # Walk through subfolders of xp
    for root, dirs, files in os.walk(xp_root):
        if "history.json" in files:
            exp_name = os.path.basename(root)  # subfolder name = experiment signature
            history_path = os.path.join(root, "history.json")

            try:
                with open(history_path, "r") as f:
                    history = json.load(f)

                if isinstance(history, list) and len(history) > 0:
                    metrics = history[-1]  # last entry (assuming chronological order)
                    results.append({
                        "Experiment": exp_name,
                        "Precision": metrics.get("precision"),
                        "Recall": metrics.get("recall"),
                        "F1": metrics.get("f1")
                    })
            except Exception as e:
                print(f"⚠️ Could not read {history_path}: {e}")

    # Convert to DataFrame
    df = pd.DataFrame(results)

    if not df.empty:
        df = df.sort_values(by="F1", ascending=False).reset_index(drop=True)

    return df


if __name__ == "__main__":
    df = aggregate_histories("outputs/xps")
    if df.empty:
        print("No histories found in xps/")
    else:
        # Print pretty table
        print(tabulate(df, headers="keys", tablefmt="pretty", floatfmt=".4f"))
        
        # Optionally save to CSV
        df.to_csv("aggregated_metrics.csv", index=False)
        print("\n✅ Aggregated results saved to aggregated_metrics.csv")
