from ultralytics import YOLO
from pathlib import Path
import shutil

if __name__ == "__main__":
    yaml_path = Path("data/overwatch.yaml")
    model = YOLO("yolov8n.pt")

    print("Avvio training smoke test (3 epoch)...")
    project_dir = str(Path(__file__).parent / "runs")
    results = model.train(
        data=str(yaml_path),
        epochs=3,
        batch=16,
        imgsz=640,
        device="0",
        freeze=10,
        workers=4,
        project=project_dir,
        name="smoke_test",
        exist_ok=True,
        verbose=False,
    )

    key = "metrics/mAP50(B)"
    val = results.results_dict.get(key, "N/A")
    print(f"\nTraining completato.")
    print(f"mAP@0.5 (dataset sintetico): {val}")

    best = Path(project_dir) / "smoke_test" / "weights" / "best.pt"
    if best.exists():
        shutil.copy(best, Path(__file__).parent / "weights" / "ow_detector.pt")
        print(f"Modello copiato in weights/ow_detector.pt")
