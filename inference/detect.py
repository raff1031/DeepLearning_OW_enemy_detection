"""
Detector — inferenza real-time con YOLOv8 + screen capture.

Gira in un thread separato e pubblica i risultati in una Queue
che overlay.py consuma per disegnare gli ESP.

Uso standalone (senza overlay, solo statistiche):
  python inference/detect.py
"""

import os
import sys
import json
import time
import threading
import queue
from pathlib import Path
from dataclasses import dataclass, field

import cv2
import numpy as np
from typing import Optional, List

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.capture import ScreenCapture

ROOT          = Path(__file__).parent.parent
DEFAULT_MODEL = ROOT / "weights" / "ow_detector.pt"
ALLY_HSV_CFG  = ROOT / "ally_hsv_config.json"

FALLBACK_MODEL = "yolov8n.pt"


@dataclass
class Detection:
    """Singola rilevazione da un frame."""
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    class_id: int
    label: str
    track_id: int = -1   # ByteTrack ID stabile tra frame, -1 = non assegnato


@dataclass
class FrameResult:
    """Risultato completo di un frame."""
    detections: List[Detection] = field(default_factory=list)
    fps: float = 0.0
    frame_w: int = 1920
    frame_h: int = 1080


class Detector:
    """
    Detector real-time thread-safe.
    Cattura lo schermo, esegue inferenza YOLO e pubblica risultati.
    """

    CLASS_NAMES = {0: "enemy", 1: "ally"}

    def __init__(
        self,
        model_path: str = None,
        conf_threshold: float = 0.30,
        iou_threshold:  float = 0.45,
        imgsz: int      = 1280,
        device: str     = "0",
    ):
        from ultralytics import YOLO

        if model_path is None:
            model_path = str(DEFAULT_MODEL) if DEFAULT_MODEL.exists() else FALLBACK_MODEL

        print(f"[Detector] Carico modello: {model_path}")
        self.model = YOLO(model_path)
        self.conf  = conf_threshold
        self.iou   = iou_threshold
        self.imgsz = imgsz
        self.device = device

        self.capture = ScreenCapture()

        self._result_queue: "queue.Queue[FrameResult]" = queue.Queue(maxsize=2)
        self._stop_event   = threading.Event()
        self._thread       = None

        # FPS tracking
        self._fps_window      = []
        self._fps_window_size = 30

        # Kernel per HSV ally detection
        self._hsv_kernel = np.ones((5, 5), np.uint8)

        # Config HSV alleati (opzionale — richiede ally_hsv_config.json)
        self._ally_hsv: Optional[dict] = None
        if ALLY_HSV_CFG.exists():
            with open(ALLY_HSV_CFG) as f:
                self._ally_hsv = json.load(f)
            print(f"[Detector] Ally HSV config caricata ({ALLY_HSV_CFG.name})")

    def start(self) -> None:
        """Avvia il thread di inferenza in background."""
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._inference_loop, daemon=True)
        self._thread.start()
        print("[Detector] Thread inferenza avviato.")

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=3)
        print("[Detector] Fermato.")

    def get_latest(self) -> Optional[FrameResult]:
        """Ritorna l'ultimo risultato disponibile (non bloccante)."""
        result = None
        while not self._result_queue.empty():
            try:
                result = self._result_queue.get_nowait()
            except queue.Empty:
                break
        return result

    def _detect_allies_hsv(self, frame: np.ndarray, fw: int, fh: int) -> List[Detection]:
        """Rileva alleati tramite HSV masking. Richiede ally_hsv_config.json."""
        cfg  = self._ally_hsv
        hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lo   = np.array([cfg["h_min"], cfg["s_min"], cfg["v_min"]])
        hi   = np.array([cfg["h_max"], cfg["s_max"], cfg["v_max"]])
        mask = cv2.inRange(hsv, lo, hi)
        if "h_min2" in cfg:
            lo2  = np.array([cfg["h_min2"], cfg["s_min"], cfg["v_min"]])
            hi2  = np.array([cfg["h_max2"], cfg["s_max"], cfg["v_max"]])
            mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lo2, hi2))
        mask = cv2.dilate(mask, self._hsv_kernel, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        allies = []
        for cnt in contours:
            if cv2.contourArea(cnt) < cfg.get("min_area", 400):
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            if not (0.25 <= (h / w if w > 0 else 0) <= 3.5):
                continue
            if x < 2 or y < 2 or (x + w) > fw - 2 or (y + h) > fh - 2:
                continue
            allies.append(Detection(x, y, x + w, y + h, 0.85, 1, "ally", track_id=-1))
        return allies

    def _inference_loop(self) -> None:
        while not self._stop_event.is_set():
            t0 = time.perf_counter()

            frame = self.capture.get_frame()
            fh, fw = frame.shape[:2]

            # ByteTrack: persist=True mantiene lo stato del tracker tra frame consecutivi
            try:
                raw_results = self.model.track(
                    source  = frame,
                    conf    = self.conf,
                    iou     = self.iou,
                    imgsz   = self.imgsz,
                    device  = self.device,
                    verbose = False,
                    persist = True,
                    tracker = "bytetrack.yaml",
                )
            except Exception:
                raw_results = self.model.predict(
                    source  = frame,
                    conf    = self.conf,
                    iou     = self.iou,
                    imgsz   = self.imgsz,
                    device  = self.device,
                    verbose = False,
                )

            detections = []
            for r in raw_results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf     = float(box.conf[0])
                    cls_id   = int(box.cls[0])
                    label    = self.CLASS_NAMES.get(cls_id, f"cls{cls_id}")
                    track_id = int(box.id[0]) if box.id is not None else -1
                    detections.append(Detection(x1, y1, x2, y2, conf, cls_id, label, track_id))

            # Aggiungi alleati rilevati via HSV (se ally_hsv_config.json presente)
            if self._ally_hsv is not None:
                detections.extend(self._detect_allies_hsv(frame, fw, fh))

            dt = time.perf_counter() - t0
            self._fps_window.append(1.0 / dt if dt > 0 else 0)
            if len(self._fps_window) > self._fps_window_size:
                self._fps_window.pop(0)
            fps = sum(self._fps_window) / len(self._fps_window)

            result = FrameResult(detections=detections, fps=fps, frame_w=fw, frame_h=fh)

            # Svuota la queue se piena (preferiamo sempre il frame piu' recente)
            if self._result_queue.full():
                try:
                    self._result_queue.get_nowait()
                except queue.Empty:
                    pass
            self._result_queue.put(result)


def run_standalone(conf: float = 0.45, device: str = "0") -> None:
    """
    Modalita' standalone: mostra i rilevamenti in una finestra OpenCV
    senza overlay separato. Utile per debug e valutazione rapida.
    """
    detector = Detector(conf_threshold=conf, device=device)
    capture  = ScreenCapture()
    detector.start()

    print("[Detect] Avviato. Premi 'Q' per uscire.")

    while True:
        frame  = capture.get_frame()
        result = detector.get_latest()

        if result:
            for det in result.detections:
                color = (0, 255, 0) if det.label == "enemy" else (255, 200, 0)
                cv2.rectangle(frame, (det.x1, det.y1), (det.x2, det.y2), color, 2)
                label_txt = f"{det.label} {det.confidence:.2f}"
                cv2.putText(frame, label_txt, (det.x1, det.y1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            cv2.putText(frame, f"FPS: {result.fps:.1f}", (15, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 255), 2)

        scale  = 960 / frame.shape[1]
        disp_h = int(frame.shape[0] * scale)
        cv2.imshow("OW Detector", cv2.resize(frame, (960, disp_h)))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    detector.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf",   type=float, default=0.45)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()
    run_standalone(conf=args.conf, device=args.device)
