import argparse
from pathlib import Path
import yaml
from ultralytics import YOLO

def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    project_dir = Path(cfg["project"])
    project_dir.mkdir(parents=True, exist_ok=True)

    for seed in cfg["seeds"]:
        print(f"\n========== Training seed {seed} ==========")
        model = YOLO(cfg["model"])
        run_name = f"seed_{seed}"

        model.train(
            data=cfg["data"],
            imgsz=cfg["imgsz"],
            epochs=cfg["epochs"],
            batch=cfg["batch"],
            seed=seed,
            project=str(project_dir),
            name=run_name,
            exist_ok=True,
            device=cfg["device"],
            workers=cfg["workers"],
            verbose=True
        )

        try:
            save_dir = Path(model.trainer.save_dir)
        except Exception:
            save_dir = project_dir / run_name

        best_pt = save_dir / "weights" / "best.pt"
        print(f"[OK] Seed {seed} finished. best.pt: {best_pt}")

    print("\nAll seeds completed.")

if __name__ == "__main__":
    main()
