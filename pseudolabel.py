"""
Pseudo-labeling pipeline:
1. Cattura screenshot OW ogni 10s per 20 minuti
2. Usa YOLOv8n (pre-trainato COCO) per rilevare persone/personaggi
3. Salva frame + label YOLO solo con confidence > soglia
4. Risultato: dataset OW annotato automaticamente, zero API key

Poi si fa fine-tuning su questi dati per specializzarsi su OW.
"""
import cv2, time, os, json
from pathlib import Path
from ultralytics import YOLO

ROOT     = Path(__file__).parent
IMG_DIR  = ROOT / "datasets" / "images" / "raw"
LBL_DIR  = ROOT / "datasets" / "labels" / "raw"
IMG_DIR.mkdir(parents=True, exist_ok=True)
LBL_DIR.mkdir(parents=True, exist_ok=True)

INTERVAL  = 10        # secondi tra uno screenshot e l'altro
DURATION  = 20 * 60  # 20 minuti
CONF      = 0.35      # soglia confidence (bassa perche' filtriamo dopo)
PERSON_ID = 0         # classe "person" in COCO

# Classi COCO che possono essere personaggi OW
VALID_CLASSES = {0: "person"}


def capture_screen():
    from utils.capture import ScreenCapture
    cap = ScreenCapture()
    return cap.get_frame()


def run(interval=INTERVAL, duration=DURATION, conf=CONF):
    print("[PseudoLabel] Carico YOLOv8n pre-trainato su COCO...")
    model = YOLO("yolov8n.pt")

    existing  = len(list(IMG_DIR.glob("*.jpg")))
    frame_id  = existing
    saved     = 0
    skipped   = 0
    t_start   = time.time()
    t_next    = t_start

    print(f"[PseudoLabel] Avvio — {duration//60} min, ogni {interval}s, conf>{conf}")
    print(f"[PseudoLabel] Frame gia' presenti: {existing}")
    print("[PseudoLabel] Ctrl+C per fermare\n")

    try:
        while (time.time() - t_start) < duration:
            now = time.time()
            if now < t_next:
                time.sleep(0.5)
                continue

            t_next    = now + interval
            elapsed   = now - t_start
            remaining = duration - elapsed

            frame = capture_screen()
            fh, fw = frame.shape[:2]

            results = model.predict(
                source=frame, conf=conf, verbose=False,
                classes=list(VALID_CLASSES.keys())
            )

            boxes = []
            for r in results:
                for box in r.boxes:
                    cls_id  = int(box.cls[0])
                    conf_v  = float(box.conf[0])
                    if cls_id not in VALID_CLASSES:
                        continue
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    bw, bh = x2 - x1, y2 - y1
                    # Filtra box troppo piccoli o fuori schermo
                    if bw < 20 or bh < 20:
                        continue
                    ratio = bh / bw if bw > 0 else 0
                    if not (0.5 <= ratio <= 5.0):
                        continue
                    cx = (x1 + bw/2) / fw
                    cy = (y1 + bh/2) / fh
                    nw = bw / fw
                    nh = bh / fh
                    boxes.append((cx, cy, nw, nh, conf_v))

            if not boxes:
                skipped += 1
                print(f"  [{int(elapsed):>4}s] SKIP  skip={skipped}  "
                      f"rimasti {int(remaining//60)}m{int(remaining%60):02d}s")
                continue

            img_path = IMG_DIR / f"{frame_id:06d}.jpg"
            lbl_path = LBL_DIR / f"{frame_id:06d}.txt"
            cv2.imwrite(str(img_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            with open(lbl_path, "w") as f:
                for (cx, cy, nw, nh, _) in boxes:
                    f.write(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")

            saved    += 1
            frame_id += 1
            confs     = [f"{b[4]:.2f}" for b in boxes]
            print(f"  [{int(elapsed):>4}s] SALVATO #{frame_id:04d}  "
                  f"persone={len(boxes)}  conf={confs}  "
                  f"rimasti {int(remaining//60)}m{int(remaining%60):02d}s")

    except KeyboardInterrupt:
        print("\n[PseudoLabel] Interrotto.")

    total = time.time() - t_start
    print(f"\n[PseudoLabel] Completato: {saved} frame salvati, {skipped} saltati")
    print(f"  Dataset totale: {frame_id} immagini")
    print(f"  Prossimo step: python main.py --mode dataset --augment")
    print(f"  Poi:           python main.py --mode train --epochs 100")


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--interval", type=int,   default=10)
    p.add_argument("--duration", type=int,   default=1200, help="secondi totali")
    p.add_argument("--conf",     type=float, default=0.35)
    args = p.parse_args()
    run(interval=args.interval, duration=args.duration, conf=args.conf)
