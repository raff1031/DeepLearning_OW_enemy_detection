"""
ESP Overlay — finestra PyQt5 trasparente sempre-in-primo-piano.

Tecnica:
  - FramelessWindowHint: nessun bordo/titolo
  - WindowStaysOnTopHint: sopra il gioco
  - WA_TranslucentBackground: sfondo completamente trasparente
  - WindowTransparentForInput: i click passano attraverso alla finestra di gioco

Il Detector gira in un thread separato e pubblica FrameResult in una Queue.
Il QTimer dell'overlay interroga la Queue ogni ~16ms (60 FPS) e ridisegna.

Uso:
  python inference/overlay.py
  (o tramite main.py --mode overlay)

Hotkey:
  F9  — toggle visibilita' overlay
  F10 — esci
"""

import os
import sys
import math
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget
from PyQt5.QtCore    import Qt, QTimer, QRect, QPoint
from PyQt5.QtGui     import QPainter, QColor, QPen, QFont, QBrush

from typing import List, Optional, Tuple
from inference.detect import Detector, FrameResult, Detection


# ─── Palette colori ESP ───────────────────────────────────────────────────────

ESP_COLORS = {
    "enemy": QColor(255, 60,  60,  220),   # rosso
    "ally":  QColor(60,  200, 255, 220),   # azzurro
    "default": QColor(255, 220, 0,  220),  # giallo
}

AIM_COLOR       = QColor(255, 165,  0, 240)   # arancione — dot sul target
AIM_LINE_COLOR  = QColor(255, 165,  0, 120)   # arancione semitrasparente — linea
AIM_LOCK_COLOR  = QColor(255,  50, 50, 240)   # rosso — quando sei già puntato

BOX_THICKNESS   = 2
CORNER_LEN      = 12    # lunghezza degli angoli decorativi del box
LABEL_FONT_SIZE = 10
CONF_BAR_H      = 4     # altezza barra confidence sotto il box
AIM_DOT_RADIUS  = 6     # raggio dot aim indicator
AIM_LOCK_DIST   = 30    # pixel: entro questa distanza il dot diventa rosso
AIM_HEAD_RATIO  = 0.30  # posizione verticale target nel box (0=top, 1=bottom)
AIM_SMOOTH      = 0.12  # EMA smoothing (0=rigido, 1=nessuno smoothing)
AIM_SWITCH_MARGIN = 80  # pixel: cambia target solo se il nuovo è più vicino di questa soglia


# ─── Widget di disegno ────────────────────────────────────────────────────────

class ESPWidget(QWidget):
    """
    Widget trasparente che disegna i box ESP.
    Riceve i dati tramite update_detections().
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self._detections: List[Detection] = []
        self._fps: float = 0.0
        self._visible_esp = True
        self._visible_aim = True
        self._screen_cx: int = 960   # aggiornato da OverlayWindow
        self._screen_cy: int = 540
        # Stato aim indicator
        self._aim_smooth_x: float = 960.0
        self._aim_smooth_y: float = 540.0
        self._aim_locked_id: int = -1   # indice del target corrente
        self._aim_initialized: bool = False

    def update_detections(self, result: "FrameResult") -> None:
        self._detections = result.detections
        self._fps        = result.fps
        self.update()

    def toggle_esp(self) -> None:
        self._visible_esp = not self._visible_esp
        self.update()

    def toggle_aim(self) -> None:
        self._visible_aim = not self._visible_aim
        self.update()

    def _nearest_enemy(self) -> Optional[Tuple[float, float, float]]:
        """
        Restituisce (smooth_x, smooth_y, distanza_raw) del nemico più vicino.
        Applica isteresi per non saltare tra target e EMA per smoothing posizione.
        """
        if not self._detections:
            self._aim_initialized = False
            self._aim_locked_id = -1
            return None

        cx, cy = self._screen_cx, self._screen_cy

        # Calcola raw target per ogni detection
        targets = []
        for i, det in enumerate(self._detections):
            tx = (det.x1 + det.x2) / 2
            ty = det.y1 + (det.y2 - det.y1) * AIM_HEAD_RATIO
            dist = math.hypot(tx - cx, ty - cy)
            targets.append((i, tx, ty, dist))

        # Isteresi: mantieni il target corrente a meno che un altro sia più vicino di AIM_SWITCH_MARGIN
        locked_idx = self._aim_locked_id
        locked_valid = 0 <= locked_idx < len(targets)

        if locked_valid:
            locked_dist = targets[locked_idx][3]
            best_idx = min(range(len(targets)), key=lambda i: targets[i][3])
            best_dist = targets[best_idx][3]
            # Cambia solo se il nuovo è significativamente più vicino
            if best_dist < locked_dist - AIM_SWITCH_MARGIN:
                self._aim_locked_id = best_idx
        else:
            # Nessun target corrente: scegli il più vicino
            self._aim_locked_id = min(range(len(targets)), key=lambda i: targets[i][3])

        _, tx, ty, dist = targets[self._aim_locked_id]

        # EMA smoothing sulla posizione
        if not self._aim_initialized:
            self._aim_smooth_x = tx
            self._aim_smooth_y = ty
            self._aim_initialized = True
        else:
            self._aim_smooth_x += AIM_SMOOTH * (tx - self._aim_smooth_x)
            self._aim_smooth_y += AIM_SMOOTH * (ty - self._aim_smooth_y)

        return self._aim_smooth_x, self._aim_smooth_y, dist

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # FPS counter sempre visibile
        self._draw_fps(painter)

        if not self._visible_esp:
            painter.end()
            return

        for det in self._detections:
            color = ESP_COLORS.get(det.label, ESP_COLORS["default"])
            self._draw_box(painter, det, color)
            self._draw_label(painter, det, color)
            self._draw_conf_bar(painter, det, color)

        if self._visible_aim:
            self._draw_aim_indicator(painter)

        painter.end()

    def _draw_box(self, p: QPainter, det: Detection, color: QColor) -> None:
        """Disegna box con angoli decorativi invece dei lati completi."""
        pen = QPen(color, BOX_THICKNESS)
        p.setPen(pen)

        x1, y1, x2, y2 = det.x1, det.y1, det.x2, det.y2
        cl = CORNER_LEN

        # Angolo top-left
        p.drawLine(x1, y1, x1 + cl, y1)
        p.drawLine(x1, y1, x1, y1 + cl)
        # Angolo top-right
        p.drawLine(x2 - cl, y1, x2, y1)
        p.drawLine(x2, y1, x2, y1 + cl)
        # Angolo bottom-left
        p.drawLine(x1, y2 - cl, x1, y2)
        p.drawLine(x1, y2, x1 + cl, y2)
        # Angolo bottom-right
        p.drawLine(x2, y2 - cl, x2, y2)
        p.drawLine(x2 - cl, y2, x2, y2)

        # Lati centrali semitrasparenti
        fade = QColor(color)
        fade.setAlpha(80)
        p.setPen(QPen(fade, 1))
        mid = cl + 4
        p.drawLine(x1 + mid, y1, x2 - mid, y1)
        p.drawLine(x1 + mid, y2, x2 - mid, y2)
        p.drawLine(x1, y1 + mid, x1, y2 - mid)
        p.drawLine(x2, y1 + mid, x2, y2 - mid)

    def _draw_label(self, p: QPainter, det: Detection, color: QColor) -> None:
        font = QFont("Consolas", LABEL_FONT_SIZE, QFont.Bold)
        p.setFont(font)

        txt = f"{det.label.upper()}  {det.confidence * 100:.0f}%"

        # Sfondo etichetta
        fm    = p.fontMetrics()
        tw    = fm.horizontalAdvance(txt)
        th    = fm.height()
        lx    = det.x1
        ly    = det.y1 - th - 6

        bg = QColor(0, 0, 0, 150)
        p.fillRect(lx, ly, tw + 8, th + 4, bg)

        p.setPen(QPen(color))
        p.drawText(lx + 4, ly + th, txt)

    def _draw_conf_bar(self, p: QPainter, det: Detection, color: QColor) -> None:
        """Barra di confidenza sotto il box."""
        bw = det.x2 - det.x1
        bx = det.x1
        by = det.y2 + 3

        # Sfondo grigio
        p.fillRect(bx, by, bw, CONF_BAR_H, QColor(60, 60, 60, 160))
        # Riempimento proporzionale alla confidence
        fill_w = int(bw * det.confidence)
        p.fillRect(bx, by, fill_w, CONF_BAR_H, color)

    def _draw_aim_indicator(self, p: QPainter) -> None:
        """Linea + dot sull'area testa del nemico più vicino al crosshair."""
        target = self._nearest_enemy()
        if target is None:
            return

        sx, sy, dist = target
        tx, ty = int(sx), int(sy)
        cx, cy = self._screen_cx, self._screen_cy
        locked = dist <= AIM_LOCK_DIST

        dot_color  = AIM_LOCK_COLOR if locked else AIM_COLOR
        line_color = QColor(dot_color)
        line_color.setAlpha(80)

        # Linea tratteggiata dal crosshair al target
        p.setPen(QPen(line_color, 1, Qt.DashLine))
        p.drawLine(cx, cy, tx, ty)

        # Dot pieno sul target
        p.setPen(QPen(dot_color, 2))
        p.setBrush(QBrush(dot_color))
        p.drawEllipse(tx - AIM_DOT_RADIUS, ty - AIM_DOT_RADIUS,
                      AIM_DOT_RADIUS * 2, AIM_DOT_RADIUS * 2)
        p.setBrush(Qt.NoBrush)

        # Cerchio esterno
        outer_color = QColor(dot_color)
        outer_color.setAlpha(55)
        p.setPen(QPen(outer_color, 1))
        p.drawEllipse(tx - 14, ty - 14, 28, 28)

        # Distanza + delta
        dx = tx - cx
        dy = ty - cy
        label = f"{int(dist)}px  ({'+' if dx>=0 else ''}{dx}, {'+' if dy>=0 else ''}{dy})"
        font = QFont("Consolas", 9, QFont.Bold)
        p.setFont(font)
        p.setPen(QPen(dot_color))
        p.drawText(tx + 14, ty + 4, label)

    def _draw_fps(self, p: QPainter) -> None:
        font = QFont("Consolas", 13, QFont.Bold)
        p.setFont(font)
        txt = f"OW-DETECTOR  {self._fps:.1f} FPS"
        p.setPen(QPen(QColor(0, 255, 120, 230)))
        p.drawText(QPoint(20, 40), txt)

        font2 = QFont("Consolas", 10)
        p.setFont(font2)

        status_esp = "ESP ON" if self._visible_esp else "ESP OFF"
        color_esp  = QColor(0, 255, 120, 180) if self._visible_esp else QColor(255, 80, 80, 180)
        p.setPen(QPen(color_esp))
        p.drawText(QPoint(20, 60), f"{status_esp}  [F9]")

        status_aim = "AIM ON" if self._visible_aim else "AIM OFF"
        color_aim  = QColor(255, 165, 0, 200) if self._visible_aim else QColor(255, 80, 80, 180)
        p.setPen(QPen(color_aim))
        p.drawText(QPoint(20, 78), f"{status_aim}  [F8]  |  F10 exit")


# ─── Finestra principale overlay ──────────────────────────────────────────────

class OverlayWindow(QMainWindow):

    def __init__(self, detector: Detector, screen_w: int = 1920, screen_h: int = 1080):
        super().__init__()
        self.detector = detector
        self._screen_w = screen_w
        self._screen_h = screen_h

        # Flags per finestra trasparente sempre in primo piano
        self.setWindowFlags(
            Qt.FramelessWindowHint       |
            Qt.WindowStaysOnTopHint      |
            Qt.Tool                      |   # nasconde dalla taskbar
            Qt.WindowTransparentForInput |   # click passano al gioco
        Qt.X11BypassWindowManagerHint
        )
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_NoSystemBackground)
        self.setGeometry(0, 0, screen_w, screen_h)

        self.esp_widget = ESPWidget(self)
        self.esp_widget.setGeometry(0, 0, screen_w, screen_h)
        self.esp_widget._screen_cx = screen_w // 2
        self.esp_widget._screen_cy = screen_h // 2
        self.setCentralWidget(self.esp_widget)

        # Timer aggiornamento ~60 FPS
        self.timer = QTimer()
        self.timer.timeout.connect(self._tick)
        self.timer.start(16)   # ms

    def _tick(self) -> None:
        result = self.detector.get_latest()
        if result:
            self.esp_widget.update_detections(result)

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key_F9:
            self.esp_widget.toggle_esp()
        elif event.key() == Qt.Key_F8:
            self.esp_widget.toggle_aim()
        elif event.key() == Qt.Key_F10:
            self.close()


# ─── Entry point ──────────────────────────────────────────────────────────────

def _find_game_monitor() -> tuple:
    """
    Trova il monitor su cui gira OW cercando il processo e la sua finestra.
    Fallback al monitor primario se non trovato.
    """
    import mss, ctypes

    # Prova a trovare la finestra di OW e il monitor su cui si trova
    try:
        import subprocess
        result = subprocess.run(
            ["powershell", "-Command",
             "Get-Process Overwatch -ErrorAction SilentlyContinue | Select-Object -ExpandProperty MainWindowHandle"],
            capture_output=True, text=True
        )
        hwnd_str = result.stdout.strip()
        if hwnd_str and hwnd_str != "0":
            hwnd = int(hwnd_str)
            # Ottieni il monitor associato alla finestra di OW
            MONITOR_DEFAULTTONEAREST = 2
            hmon = ctypes.windll.user32.MonitorFromWindow(hwnd, MONITOR_DEFAULTTONEAREST)
            info = ctypes.create_string_buffer(40)
            ctypes.c_uint32.from_buffer(info, 0).value = 40  # cbSize
            if ctypes.windll.user32.GetMonitorInfoA(hmon, info):
                # rcMonitor: left, top, right, bottom (4 x int32 a offset 4)
                import struct
                l, t, r, b = struct.unpack_from("iiii", info, 4)
                return r - l, b - t, l, t
    except Exception:
        pass

    # Fallback: usa monitor primario
    with mss.mss() as sct:
        mon = sct.monitors[1]
        return mon["width"], mon["height"], mon["left"], mon["top"]


def run_overlay(conf: float = 0.30, device: str = "0") -> None:
    # Rileva risoluzione e posizione del monitor su cui gira OW
    screen_w, screen_h, screen_x, screen_y = _find_game_monitor()
    print(f"[Overlay] Monitor rilevato: {screen_w}x{screen_h} @ ({screen_x},{screen_y})")

    detector = Detector(conf_threshold=conf, device=device)
    detector.start()

    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(True)

    window = OverlayWindow(detector, screen_w=screen_w, screen_h=screen_h)
    # Posiziona l'overlay esattamente sul monitor del gioco
    window.setGeometry(screen_x, screen_y, screen_w, screen_h)
    window.esp_widget.setGeometry(0, 0, screen_w, screen_h)
    window.show()

    print(f"[Overlay] Avviato su {screen_w}x{screen_h}")
    print("[Overlay] F9 = toggle ESP | F10 = esci")

    try:
        app.exec_()
    finally:
        detector.stop()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf",   type=float, default=0.45)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()
    run_overlay(conf=args.conf, device=args.device)
