import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_threshold_curve(csv_path, out_path, title="Threshold sweep"):
    df = pd.read_csv(csv_path)
    plt.figure(figsize=(7, 5))
    sns.lineplot(data=df, x="conf_thr", y="F1", marker="o", label="F1")
    sns.lineplot(data=df, x="conf_thr", y="Precision", marker="o", label="Precision")
    sns.lineplot(data=df, x="conf_thr", y="Recall", marker="o", label="Recall")
    plt.xlabel("Confidence threshold")
    plt.ylabel("Metric value")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--title", type=str, default="Threshold sweep")
    args = parser.parse_args()

    plot_threshold_curve(args.csv, args.out, args.title)

if __name__ == "__main__":
    main()
