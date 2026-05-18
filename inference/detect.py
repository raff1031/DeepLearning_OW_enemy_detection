"""
Detector — inferenza real-time con YOLOv8 + screen capture.

Gira in un thread separato e pubblica i risultati in una Queue
che overlay.py consuma per disegnare gli ESP.

Uso standalone (senza overlay, solo statistiche):
  python inference/detect.py
"""

import os
import sys
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

ROOT         = Path(__file__).parent.parent
DEFAULT_MODEL = ROOT / "weights" / "ow_detector.pt"

# Fallback: se il modello custom non esiste ancora usa yolov8n pre-trainato
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
        self._fps_window   = []
        self._fps_window_size = 30

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

    def _inference_loop(self) -> None:
        while not self._stop_event.is_set():
            t0 = time.perf_counter()

            frame = self.capture.get_frame()
            fh, fw = frame.shape[:2]

            raw_results = self.model.predict(
                source     = frame,
                conf       = self.conf,
                iou        = self.iou,
                imgsz      = self.imgsz,
                device     = self.device,
                verbose    = False,
                stream     = False,
            )

            detections = []
            for r in raw_results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf    = float(box.conf[0])
                    cls_id  = int(box.cls[0])
                    label   = self.CLASS_NAMES.get(cls_id, f"cls{cls_id}")
                    detections.append(Detection(x1, y1, x2, y2, conf, cls_id, label))

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
