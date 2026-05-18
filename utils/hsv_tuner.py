"""
HSV Tuner — calibra i range HSV dell'outline nemici in Overwatch.

Istruzioni:
  1. Avvia Overwatch, vai in una partita con nemici visibili a schermo
  2. Lancia questo script in un altro terminale
  3. Usa le trackbar per isolare il colore dell'outline
  4. Premi 'S' per salvare la config in hsv_config.json
  5. Premi 'Q' per uscire

In OW2: Opzioni → Accessibilità → Colori → imposta outline nemici su
un colore solido e saturo (viola consigliato, raramente presente nell'ambiente).
"""

import cv2
import numpy as np
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.capture import ScreenCapture

CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "hsv_config.json")

PRESETS = {
    "arancione": {"h_min": 5,   "h_max": 20,  "s_min": 150, "s_max": 255, "v_min": 150, "v_max": 255, "min_area": 400},
    "viola":     {"h_min": 125, "h_max": 155, "s_min": 140, "s_max": 255, "v_min": 80,  "v_max": 255, "min_area": 400},
    "rosso":     {"h_min": 0,   "h_max": 10,  "h_min2": 165, "h_max2": 180, "s_min": 150, "s_max": 255, "v_min": 100, "v_max": 255, "min_area": 400},
    "ciano":     {"h_min": 85,  "h_max": 100, "s_min": 150, "s_max": 255, "v_min": 150, "v_max": 255, "min_area": 400},
}


def nothing(_): pass


def run_tuner(preset: str = "viola"):
    cap = ScreenCapture()
    p = PRESETS.get(preset, PRESETS["viola"])

    cv2.namedWindow("HSV Tuner", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Mask", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Result", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("HSV Tuner", 640, 200)

    cv2.createTrackbar("H Min", "HSV Tuner", p["h_min"], 179, nothing)
    cv2.createTrackbar("H Max", "HSV Tuner", p["h_max"], 179, nothing)
    cv2.createTrackbar("S Min", "HSV Tuner", p["s_min"], 255, nothing)
    cv2.createTrackbar("S Max", "HSV Tuner", p["s_max"], 255, nothing)
    cv2.createTrackbar("V Min", "HSV Tuner", p["v_min"], 255, nothing)
    cv2.createTrackbar("V Max", "HSV Tuner", p["v_max"], 255, nothing)
    cv2.createTrackbar("Min Area", "HSV Tuner", 500, 5000, nothing)

    print(f"[HSV Tuner] Preset '{preset}' caricato. S=salva, Q=esci.")

    while True:
        frame = cap.get_game_region()
        hsv   = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        h_min = cv2.getTrackbarPos("H Min", "HSV Tuner")
        h_max = cv2.getTrackbarPos("H Max", "HSV Tuner")
        s_min = cv2.getTrackbarPos("S Min", "HSV Tuner")
        s_max = cv2.getTrackbarPos("S Max", "HSV Tuner")
        v_min = cv2.getTrackbarPos("V Min", "HSV Tuner")
        v_max = cv2.getTrackbarPos("V Max", "HSV Tuner")
        min_area = cv2.getTrackbarPos("Min Area", "HSV Tuner")

        lo   = np.array([h_min, s_min, v_min])
        hi   = np.array([h_max, s_max, v_max])
        mask = cv2.inRange(hsv, lo, hi)

        kernel = np.ones((5, 5), np.uint8)
        mask   = cv2.dilate(mask, kernel, iterations=2)

        result = cv2.bitwise_and(frame, frame, mask=mask)

        # Disegna i bounding box rilevati
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        preview = frame.copy()
        count = 0
        for cnt in contours:
            if cv2.contourArea(cnt) >= min_area:
                x, y, w, h = cv2.boundingRect(cnt)
                ratio = h / w if w > 0 else 0
                if 0.3 < ratio < 3.0:   # filtra box con proporzioni irreali
                    cv2.rectangle(preview, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    count += 1

        cv2.putText(preview, f"Nemici: {count}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        # Scala per visualizzazione
        disp_w = 960
        scale  = disp_w / frame.shape[1]
        disp_h = int(frame.shape[0] * scale)
        cv2.imshow("HSV Tuner", cv2.resize(np.zeros((10, 10, 3), np.uint8), (640, 10)))
        cv2.imshow("Mask",   cv2.resize(mask,    (disp_w, disp_h)))
        cv2.imshow("Result", cv2.resize(preview, (disp_w, disp_h)))

        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            config = {
                "h_min": h_min, "h_max": h_max,
                "s_min": s_min, "s_max": s_max,
                "v_min": v_min, "v_max": v_max,
                "min_area": min_area,
            }
            with open(CONFIG_PATH, "w") as f:
                json.dump(config, f, indent=2)
            print(f"[HSV Tuner] Config salvata in {CONFIG_PATH}")
            print(f"  {config}")

    cv2.destroyAllWindows()


def load_config() -> dict:
    if not os.path.exists(CONFIG_PATH):
        print(f"[WARN] hsv_config.json non trovato. Usa utils/hsv_tuner.py per calibrare.")
        return PRESETS["viola"]
    with open(CONFIG_PATH) as f:
        return json.load(f)


if __name__ == "__main__":
    preset = sys.argv[1] if len(sys.argv) > 1 else "viola"
    run_tuner(preset)
