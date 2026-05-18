"""
Trainer — transfer learning da YOLOv8s pre-trainato su COCO.

Strategia:
  Fase 1 (epoch 1-30):    backbone congelato, solo detection head si allena
  Fase 2 (epoch 31-70):   sblocca ultimi 3 layer backbone (fine-tuning leggero)
  Fase 3 (epoch 71-100):  sblocca tutto con lr molto bassa (full fine-tuning)

Uso:
  python training/train.py
  python training/train.py --model yolov8n.pt --epochs 50 --batch 32
"""

import os
import sys
import argparse
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT      = Path(__file__).parent.parent
YAML_PATH = ROOT / "data" / "overwatch.yaml"
WEIGHTS   = ROOT / "weights"


def train(
    model_name: str  = "yolov8s.pt",
    epochs: int      = 100,
    batch: int       = 32,
    imgsz: int       = 640,
    device: str      = "0",       # "0" = prima GPU, "cpu" per CPU
    patience: int    = 20,        # early stopping
    workers: int     = 8,
) -> None:
    from ultralytics import YOLO

    if not YAML_PATH.exists():
        raise FileNotFoundError(
            f"{YAML_PATH} non trovato. Esegui prima: python data/dataset.py --augment"
        )

    WEIGHTS.mkdir(exist_ok=True)
    project_dir = str(ROOT / "runs")

    print(f"[Train] Modello base: {model_name}")
    print(f"[Train] Dataset:      {YAML_PATH}")
    print(f"[Train] Epochs:       {epochs}  Batch: {batch}  imgsz: {imgsz}")

    # ─── FASE 1: backbone congelato ───────────────────────────────────────────
    phase1_epochs = min(30, epochs // 3)
    print(f"\n[Train] FASE 1 — backbone congelato ({phase1_epochs} epoch)")

    model = YOLO(model_name)
    results = model.train(
        data      = str(YAML_PATH),
        epochs    = phase1_epochs,
        batch     = batch,
        imgsz     = imgsz,
        device    = device,
        workers   = workers,
        patience  = patience,
        freeze    = 10,           # congela i primi 10 moduli (backbone completo)
        lr0       = 1e-3,
        lrf       = 0.01,
        momentum  = 0.937,
        weight_decay = 0.0005,
        warmup_epochs = 3,
        project   = project_dir,
        name      = "phase1",
        exist_ok  = True,
        plots     = True,
        save      = True,
    )

    best_phase1 = Path(project_dir) / "phase1" / "weights" / "best.pt"

    # ─── FASE 2: sblocca ultimi layer ─────────────────────────────────────────
    if epochs > phase1_epochs:
        phase2_epochs = min(40, epochs // 3)
        print(f"\n[Train] FASE 2 — unfreeze parziale ({phase2_epochs} epoch)")

        model2 = YOLO(str(best_phase1))
        model2.train(
            data      = str(YAML_PATH),
            epochs    = phase2_epochs,
            batch     = batch,
            imgsz     = imgsz,
            device    = device,
            workers   = workers,
            patience  = patience,
            freeze    = 7,        # sblocca ultimi 3 moduli backbone
            lr0       = 3e-4,
            lrf       = 0.01,
            project   = project_dir,
            name      = "phase2",
            exist_ok  = True,
            plots     = True,
            save      = True,
        )
        best_phase2 = Path(project_dir) / "phase2" / "weights" / "best.pt"
    else:
        best_phase2 = best_phase1

    # ─── FASE 3: full fine-tuning ──────────────────────────────────────────────
    remaining = epochs - phase1_epochs - (phase2_epochs if epochs > phase1_epochs else 0)
    if remaining > 5:
        print(f"\n[Train] FASE 3 — full fine-tuning ({remaining} epoch, lr molto bassa)")

        model3 = YOLO(str(best_phase2))
        model3.train(
            data      = str(YAML_PATH),
            epochs    = remaining,
            batch     = batch,
            imgsz     = imgsz,
            device    = device,
            workers   = workers,
            patience  = patience,
            freeze    = 0,        # tutto sbloccato
            lr0       = 5e-5,
            lrf       = 0.01,
            project   = project_dir,
            name      = "phase3",
            exist_ok  = True,
            plots     = True,
            save      = True,
        )
        final_best = Path(project_dir) / "phase3" / "weights" / "best.pt"
    else:
        final_best = best_phase2

    # Copia il modello migliore in weights/
    import shutil
    dest = WEIGHTS / "ow_detector.pt"
    shutil.copy(final_best, dest)
    print(f"\n[Train] Modello finale salvato: {dest}")
    print("[Train] Per usarlo in inferenza: python main.py --mode detect")


def evaluate(model_path: str = None) -> None:
    """Valuta il modello sul test set e stampa le metriche."""
    from ultralytics import YOLO

    if model_path is None:
        model_path = str(WEIGHTS / "ow_detector.pt")

    model = YOLO(model_path)
    metrics = model.val(data=str(YAML_PATH), split="test")

    print("\n[Eval] Risultati sul test set:")
    print(f"  mAP@0.5:      {metrics.box.map50:.4f}")
    print(f"  mAP@0.5:0.95: {metrics.box.map:.4f}")
    print(f"  Precision:    {metrics.box.mp:.4f}")
    print(f"  Recall:       {metrics.box.mr:.4f}")


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()

    parser = argparse.ArgumentParser(description="OW Detector Training")
    parser.add_argument("--model",   default="yolov8s.pt", help="Modello base Ultralytics")
    parser.add_argument("--epochs",  type=int, default=100)
    parser.add_argument("--batch",   type=int, default=32)
    parser.add_argument("--imgsz",   type=int, default=640)
    parser.add_argument("--device",  default="0",          help="GPU id o 'cpu'")
    parser.add_argument("--eval",    action="store_true",  help="Solo valutazione")
    args = parser.parse_args()

    if args.eval:
        evaluate()
    else:
        train(
            model_name=args.model,
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            device=args.device,
        )
