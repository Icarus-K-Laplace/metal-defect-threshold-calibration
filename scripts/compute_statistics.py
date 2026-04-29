import argparse
from pathlib import Path
import pandas as pd

def mean_std(series):
    return series.mean(), series.std(ddof=1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="CSV with per-seed results")
    parser.add_argument("--output", type=str, required=True, help="Summary CSV")
    args = parser.parse_args()

    df = pd.read_csv(args.input)

    metrics = [c for c in df.columns if c not in ["seed", "weights", "save_dir"]]

    rows = []
    for col in metrics:
        if pd.api.types.is_numeric_dtype(df[col]):
            m, s = mean_std(df[col])
            rows.append({"metric": col, "mean": m, "std": s})

    out = pd.DataFrame(rows)
    out.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(out)

if __name__ == "__main__":
    main()
