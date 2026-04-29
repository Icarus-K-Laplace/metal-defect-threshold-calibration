import argparse
from pathlib import Path
import pandas as pd

def load_csv(path):
    return pd.read_csv(path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=str, required=True)
    parser.add_argument("--p2", type=str, required=True)
    parser.add_argument("--p2clahe", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    baseline = load_csv(args.baseline)
    p2 = load_csv(args.p2)
    p2clahe = load_csv(args.p2clahe)

    rows = []
    for name, df in [("baseline", baseline), ("p2", p2), ("p2_clahe", p2clahe)]:
        row = {
            "method": name,
            "precision_mean": df["test_Precision"].mean(),
            "precision_std": df["test_Precision"].std(ddof=1),
            "recall_mean": df["test_Recall"].mean(),
            "recall_std": df["test_Recall"].std(ddof=1),
            "f1_mean": df["test_F1"].mean(),
            "f1_std": df["test_F1"].std(ddof=1),
        }
        rows.append(row)

    out = pd.DataFrame(rows)
    out.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(out)

if __name__ == "__main__":
    main()
