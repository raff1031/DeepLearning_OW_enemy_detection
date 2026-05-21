"""
HSV Tuner — calibra i range HSV degli outline in Overwatch.

Uso:
  python utils/hsv_tuner.py rosso            # calibra nemici → hsv_config.json
  python utils/hsv_tuner.py ciano --ally     # calibra alleati → ally_hsv_config.json

In OW2: Opzioni → Accessibilità → Colori → imposta colori outline distinti
  per nemici e alleati (es. rosso nemici, ciano alleati).
"""

import cv2
import numpy as np
import json
import os
import sys
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.capture import ScreenCapture

ROOT = os.path.dirname(os.path.dirname(__file__))
CONFIGS = {
    "enemy": os.path.join(ROOT, "hsv_config.json"),
    "ally":  os.path.join(ROOT, "ally_hsv_config.json"),
}

PRESETS = {
    "arancione": {"h_min": 5,   "h_max": 20,  "s_min": 150, "s_max": 255, "v_min": 150, "v_max": 255, "min_area": 400},
    "viola":     {"h_min": 125, "h_max": 155, "s_min": 140, "s_max": 255, "v_min": 80,  "v_max": 255, "min_area": 400},
    "rosso":     {"h_min": 0,   "h_max": 10,  "h_min2": 165, "h_max2": 180, "s_min": 150, "s_max": 255, "v_min": 100, "v_max": 255, "min_area": 400},
    "ciano":     {"h_min": 85,  "h_max": 100, "s_min": 150, "s_max": 255, "v_min": 150, "v_max": 255, "min_area": 400},
    "blu":       {"h_min": 100, "h_max": 130, "s_min": 130, "s_max": 255, "v_min": 100, "v_max": 255, "min_area": 400},
    "verde":     {"h_min": 40,  "h_max": 80,  "s_min": 100, "s_max": 255, "v_min": 100, "v_max": 255, "min_area": 400},
}


def nothing(_): pass


def run_tuner(preset: str = "viola", target: str = "enemy"):
    cap        = ScreenCapture()
    p          = PRESETS.get(preset, PRESETS["viola"])
    config_path = CONFIGS[target]
    label      = "ENEMY" if target == "enemy" else "ALLY"
    box_color  = (0, 80, 255) if target == "enemy" else (255, 180, 0)

    cv2.namedWindow("HSV Tuner", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Mask", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Result", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("HSV Tuner", 640, 200)

    cv2.createTrackbar("H Min",    "HSV Tuner", p["h_min"],  179,  nothing)
    cv2.createTrackbar("H Max",    "HSV Tuner", p["h_max"],  179,  nothing)
    cv2.createTrackbar("S Min",    "HSV Tuner", p["s_min"],  255,  nothing)
    cv2.createTrackbar("S Max",    "HSV Tuner", p["s_max"],  255,  nothing)
    cv2.createTrackbar("V Min",    "HSV Tuner", p["v_min"],  255,  nothing)
    cv2.createTrackbar("V Max",    "HSV Tuner", p["v_max"],  255,  nothing)
    cv2.createTrackbar("Min Area", "HSV Tuner", p.get("min_area", 400), 5000, nothing)

    print(f"[HSV Tuner] Target: {label}  |  Preset: '{preset}'")
    print(f"[HSV Tuner] Salvataggio in: {config_path}")
    print(f"[HSV Tuner] S = salva, Q = esci")

    kernel = np.ones((5, 5), np.uint8)

    while True:
        frame = cap.get_game_region()
        hsv   = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        h_min    = cv2.getTrackbarPos("H Min",    "HSV Tuner")
        h_max    = cv2.getTrackbarPos("H Max",    "HSV Tuner")
        s_min    = cv2.getTrackbarPos("S Min",    "HSV Tuner")
        s_max    = cv2.getTrackbarPos("S Max",    "HSV Tuner")
        v_min    = cv2.getTrackbarPos("V Min",    "HSV Tuner")
        v_max    = cv2.getTrackbarPos("V Max",    "HSV Tuner")
        min_area = cv2.getTrackbarPos("Min Area", "HSV Tuner")

        lo   = np.array([h_min, s_min, v_min])
        hi   = np.array([h_max, s_max, v_max])
        mask = cv2.inRange(hsv, lo, hi)
        mask = cv2.dilate(mask, kernel, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        preview = frame.copy()
        count = 0
        for cnt in contours:
            if cv2.contourArea(cnt) >= min_area:
                x, y, w, h = cv2.boundingRect(cnt)
                ratio = h / w if w > 0 else 0
                if 0.25 < ratio < 3.5:
                    cv2.rectangle(preview, (x, y), (x + w, y + h), box_color, 2)
                    count += 1

        cv2.putText(preview, f"{label}: {count}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, box_color, 2)

        disp_w = 960
        scale  = disp_w / frame.shape[1]
        disp_h = int(frame.shape[0] * scale)
        cv2.imshow("HSV Tuner", cv2.resize(np.zeros((10, 10, 3), np.uint8), (640, 10)))
        cv2.imshow("Mask",      cv2.resize(mask,    (disp_w, disp_h)))
        cv2.imshow("Result",    cv2.resize(preview, (disp_w, disp_h)))

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
            with open(config_path, "w") as f:
                json.dump(config, f, indent=2)
            print(f"[HSV Tuner] Config {label} salvata in {config_path}")

    cv2.destroyAllWindows()


def load_config(target: str = "enemy") -> dict:
    path = CONFIGS[target]
    if not os.path.exists(path):
        fallback = "viola" if target == "enemy" else "ciano"
        print(f"[WARN] {os.path.basename(path)} non trovato. Usa hsv_tuner.py --{'ally' if target == 'ally' else 'enemy'}")
        return PRESETS[fallback]
    with open(path) as f:
        return json.load(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("preset", nargs="?", default="viola",
                        choices=list(PRESETS.keys()),
                        help="Preset colore di partenza")
    parser.add_argument("--ally", action="store_true",
                        help="Calibra outline alleati (salva in ally_hsv_config.json)")
    args = parser.parse_args()

    target = "ally" if args.ally else "enemy"
    run_tuner(args.preset, target)
