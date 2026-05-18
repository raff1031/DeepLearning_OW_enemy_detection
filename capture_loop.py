"""
Cattura uno screenshot ogni 10 secondi per 20 minuti.
Applica HSV detection viola e salva frame + label YOLO solo se ci sono nemici.
"""
import cv2, numpy as np, json, os, time, sys, argparse
from utils.capture import ScreenCapture

parser = argparse.ArgumentParser()
parser.add_argument("--interval", type=int, default=10)
parser.add_argument("--duration", type=int, default=20*60)
args, _ = parser.parse_known_args()

INTERVAL   = args.interval
DURATION   = args.duration
IMG_DIR    = "datasets/images/raw"
LBL_DIR    = "datasets/labels/raw"

os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(LBL_DIR, exist_ok=True)

with open("hsv_config.json") as f:
    cfg = json.load(f)

cap     = ScreenCapture()
kernel  = np.ones((5, 5), np.uint8)

# Conta i file gia' presenti per non sovrascrivere
existing = len([f for f in os.listdir(IMG_DIR) if f.endswith(".jpg")])
frame_id = existing

t_start  = time.time()
t_next   = t_start
captured = 0
skipped  = 0

print(f"[Collector] Avvio — {DURATION//60} min, ogni {INTERVAL}s")
print(f"[Collector] Frame gia' presenti: {existing}")
print(f"[Collector] Premi Ctrl+C per fermare\n")

try:
    while (time.time() - t_start) < DURATION:
        now = time.time()
        if now < t_next:
            time.sleep(0.2)
            continue

        t_next = now + INTERVAL
        elapsed = now - t_start
        remaining = DURATION - elapsed

        frame = cap.get_frame()
        fh, fw = frame.shape[:2]

        # HSV mask (supporta doppio range per il rosso che si spezza in H=0-10 e H=165-180)
        hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lo   = np.array([cfg["h_min"],  cfg["s_min"], cfg["v_min"]])
        hi   = np.array([cfg["h_max"],  cfg["s_max"], cfg["v_max"]])
        mask = cv2.inRange(hsv, lo, hi)
        if "h_min2" in cfg:
            lo2  = np.array([cfg["h_min2"], cfg["s_min"], cfg["v_min"]])
            hi2  = np.array([cfg["h_max2"], cfg["s_max"], cfg["v_max"]])
            mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lo2, hi2))
        mask = cv2.dilate(mask, kernel, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < cfg["min_area"]:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            ratio = h / w if w > 0 else 0
            if not (0.25 <= ratio <= 3.5):
                continue
            if x < 2 or y < 2 or (x+w) > fw-2 or (y+h) > fh-2:
                continue
            boxes.append((x, y, w, h))

        if not boxes:
            skipped += 1
            print(f"  [{int(elapsed):>4}s] SKIP (nessun nemico) — rimasti {int(remaining//60)}m{int(remaining%60):02d}s")
            continue

        # Salva immagine
        img_path = os.path.join(IMG_DIR, f"{frame_id:06d}.jpg")
        lbl_path = os.path.join(LBL_DIR, f"{frame_id:06d}.txt")
        cv2.imwrite(img_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

        with open(lbl_path, "w") as f:
            for (x, y, w, h) in boxes:
                cx = (x + w/2) / fw
                cy = (y + h/2) / fh
                nw = w / fw
                nh = h / fh
                f.write(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")

        captured += 1
        frame_id += 1
        print(f"  [{int(elapsed):>4}s] SALVATO #{frame_id:04d}  nemici={len(boxes)}  "
              f"skip={skipped}  rimasti {int(remaining//60)}m{int(remaining%60):02d}s")

except KeyboardInterrupt:
    print("\n[Collector] Interrotto dall'utente.")

total = time.time() - t_start
print(f"\n[Collector] Fatto: {captured} frame salvati, {skipped} saltati, in {total/60:.1f} min")
print(f"  Dataset totale: {frame_id} immagini in {IMG_DIR}")
