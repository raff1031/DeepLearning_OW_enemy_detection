"""
Collector — pipeline di raccolta dati con auto-labeling via outline HSV.

Flusso:
  1. Cattura frame dal gioco in real-time
  2. Rileva outline nemici tramite color masking (HSV)
  3. Genera bounding box e le salva in formato YOLO (.txt)
  4. Salva il frame raw (senza overlay) come immagine di training

Uso:
  python data/collector.py --target 5000 --color viola
"""

import cv2
import numpy as np
import os
import sys
import time
import argparse
import json
from typing import List, Tuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.capture import ScreenCapture
from utils.hsv_tuner import load_config

DATASET_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "datasets")
IMAGES_DIR  = os.path.join(DATASET_DIR, "images", "raw")
LABELS_DIR  = os.path.join(DATASET_DIR, "labels", "raw")


def detect_outlines(frame: np.ndarray, cfg: dict) -> List[Tuple[int, int, int, int]]:
    """
    Trova bounding box dei nemici tramite color masking HSV.
    Ritorna lista di (x, y, w, h) in pixel.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lo   = np.array([cfg["h_min"], cfg["s_min"], cfg["v_min"]])
    hi   = np.array([cfg["h_max"], cfg["s_max"], cfg["v_max"]])
    mask = cv2.inRange(hsv, lo, hi)

    kernel = np.ones((5, 5), np.uint8)
    mask   = cv2.dilate(mask, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < cfg.get("min_area", 500):
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        ratio = h / w if w > 0 else 0
        # un personaggio in piedi ha aspect ratio ~0.3-2.5
        if not (0.3 <= ratio <= 2.5):
            continue
        # ignora box che escono dai bordi (artefatti)
        fh, fw = frame.shape[:2]
        if x < 2 or y < 2 or (x + w) > fw - 2 or (y + h) > fh - 2:
            continue
        boxes.append((x, y, w, h))

    return boxes


def save_sample(frame: np.ndarray, boxes: List, frame_id: int) -> None:
    """Salva immagine e label YOLO corrispondente."""
    os.makedirs(IMAGES_DIR, exist_ok=True)
    os.makedirs(LABELS_DIR, exist_ok=True)

    img_path = os.path.join(IMAGES_DIR, f"{frame_id:06d}.jpg")
    lbl_path = os.path.join(LABELS_DIR, f"{frame_id:06d}.txt")

    cv2.imwrite(img_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

    fh, fw = frame.shape[:2]
    with open(lbl_path, "w") as f:
        for (x, y, w, h) in boxes:
            # YOLO format: class cx cy w h  (tutto normalizzato 0-1)
            cx = (x + w / 2) / fw
            cy = (y + h / 2) / fh
            nw = w / fw
            nh = h / fh
            f.write(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")


def collect(target_frames: int = 5000, show_preview: bool = True,
            max_boxes_per_frame: int = 6, min_boxes_per_frame: int = 1) -> None:
    """
    Raccoglie frame dal gioco finche' non raggiunge target_frames frame validi.

    Args:
        target_frames: quanti frame annotati vuoi raccogliere
        show_preview:  mostra finestra OpenCV con box rilevati
        max_boxes_per_frame: scarta frame con troppi box (rumore)
        min_boxes_per_frame: scarta frame senza nemici
    """
    cfg = load_config()
    cap = ScreenCapture()

    saved   = 0
    skipped = 0
    t_start = time.time()

    print(f"[Collector] Avvio raccolta — target: {target_frames} frame")
    print(f"[Collector] Config HSV: {cfg}")
    print("[Collector] Premi 'Q' nella preview per interrompere.")

    while saved < target_frames:
        frame = cap.get_game_region()
        boxes = detect_outlines(frame, cfg)

        n = len(boxes)
        if n < min_boxes_per_frame or n > max_boxes_per_frame:
            skipped += 1
            if show_preview:
                _show_preview(frame, boxes, saved, skipped, target_frames)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            continue

        save_sample(frame, boxes, saved)
        saved += 1

        elapsed = time.time() - t_start
        fps_eff = saved / elapsed if elapsed > 0 else 0

        if saved % 100 == 0:
            eta = (target_frames - saved) / fps_eff if fps_eff > 0 else 0
            print(f"  [{saved}/{target_frames}]  skip={skipped}  "
                  f"eff={fps_eff:.1f} frame/s  ETA={eta/60:.1f} min")

        if show_preview:
            _show_preview(frame, boxes, saved, skipped, target_frames)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cv2.destroyAllWindows()
    total = time.time() - t_start
    print(f"\n[Collector] Completato: {saved} frame in {total/60:.1f} min")
    print(f"  Immagini: {IMAGES_DIR}")
    print(f"  Labels:   {LABELS_DIR}")


def _show_preview(frame: np.ndarray, boxes: list, saved: int,
                  skipped: int, target: int) -> None:
    preview = frame.copy()
    for (x, y, w, h) in boxes:
        cv2.rectangle(preview, (x, y), (x + w, y + h), (0, 255, 0), 2)

    overlay_text = [
        f"Salvati: {saved}/{target}",
        f"Scartati: {skipped}",
        f"Box ora: {len(boxes)}",
    ]
    for i, txt in enumerate(overlay_text):
        cv2.putText(preview, txt, (15, 35 + i * 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    scale   = 960 / preview.shape[1]
    disp_h  = int(preview.shape[0] * scale)
    cv2.imshow("Collector Preview", cv2.resize(preview, (960, disp_h)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OW Enemy Data Collector")
    parser.add_argument("--target",    type=int,  default=5000, help="Frame da raccogliere")
    parser.add_argument("--no-preview",action="store_true",      help="Disabilita finestra preview")
    parser.add_argument("--max-boxes", type=int,  default=6,    help="Max box per frame")
    parser.add_argument("--min-boxes", type=int,  default=1,    help="Min box per frame")
    args = parser.parse_args()

    collect(
        target_frames=args.target,
        show_preview=not args.no_preview,
        max_boxes_per_frame=args.max_boxes,
        min_boxes_per_frame=args.min_boxes,
    )
