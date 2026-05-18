"""
OW-Detector — entry point unico per tutte le modalita'.

Modalita':
  tune     →  calibra colori outline HSV (da fare PRIMA di tutto)
  collect  →  raccoglie frame annotati automaticamente
  dataset  →  prepara dataset (split + augmentation + yaml)
  train    →  traina il modello con transfer learning
  detect   →  inferenza standalone con finestra OpenCV
  overlay  →  ESP overlay trasparente sul gioco (modalita' finale)

Esempi:
  python main.py --mode tune
  python main.py --mode collect --frames 5000
  python main.py --mode dataset --augment
  python main.py --mode train --epochs 100
  python main.py --mode detect
  python main.py --mode overlay --conf 0.4
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        description="OW Enemy Detector",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--mode", required=True,
        choices=["tune", "collect", "dataset", "train", "detect", "overlay"],
        help="Modalita' di esecuzione",
    )

    # Tune
    parser.add_argument("--preset", default="viola",
                        choices=["arancione", "viola", "rosso", "ciano"],
                        help="[tune] Preset colore di partenza")

    # Collect
    parser.add_argument("--frames",    type=int,   default=5000,  help="[collect] Frame target")
    parser.add_argument("--no-preview",action="store_true",        help="[collect] Disabilita preview")
    parser.add_argument("--max-boxes", type=int,   default=6,     help="[collect] Max box per frame")

    # Dataset
    parser.add_argument("--augment",   action="store_true",        help="[dataset] Applica augmentation")
    parser.add_argument("--factor",    type=int,   default=3,     help="[dataset] Campioni aug per frame")

    # Train
    parser.add_argument("--model",     default="yolov8s.pt",      help="[train] Modello base")
    parser.add_argument("--epochs",    type=int,   default=100,   help="[train] Epoche totali")
    parser.add_argument("--batch",     type=int,   default=32,    help="[train] Batch size")
    parser.add_argument("--imgsz",     type=int,   default=640,   help="[train] Dimensione immagine")
    parser.add_argument("--eval",      action="store_true",        help="[train] Solo valutazione")

    # Detect / Overlay
    parser.add_argument("--conf",      type=float, default=0.45,  help="[detect/overlay] Confidence threshold")
    parser.add_argument("--device",    default="0",               help="GPU id ('0','1') o 'cpu'")

    args = parser.parse_args()

    # ─── Dispatch ─────────────────────────────────────────────────────────────

    if args.mode == "tune":
        print(f"[main] Avvio HSV Tuner (preset: {args.preset})")
        from utils.hsv_tuner import run_tuner
        run_tuner(preset=args.preset)

    elif args.mode == "collect":
        print(f"[main] Avvio Collector (target: {args.frames} frame)")
        from data.collector import collect
        collect(
            target_frames=args.frames,
            show_preview=not args.no_preview,
            max_boxes_per_frame=args.max_boxes,
        )

    elif args.mode == "dataset":
        print("[main] Preparazione dataset...")
        from data.dataset import prepare
        prepare(augment=args.augment, augment_factor=args.factor)

    elif args.mode == "train":
        if args.eval:
            print("[main] Valutazione modello...")
            from training.train import evaluate
            evaluate()
        else:
            print("[main] Avvio training...")
            from training.train import train
            train(
                model_name=args.model,
                epochs=args.epochs,
                batch=args.batch,
                imgsz=args.imgsz,
                device=args.device,
            )

    elif args.mode == "detect":
        print("[main] Avvio detector standalone...")
        from inference.detect import run_standalone
        run_standalone(conf=args.conf, device=args.device)

    elif args.mode == "overlay":
        print("[main] Avvio ESP overlay...")
        from inference.overlay import run_overlay
        run_overlay(conf=args.conf, device=args.device)


if __name__ == "__main__":
    main()
