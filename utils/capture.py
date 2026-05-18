import mss
import numpy as np
import cv2


class ScreenCapture:
    """
    Wrapper attorno a mss per screen capture a bassa latenza su Windows.
    mss usa la Windows DXGI Desktop Duplication API (<5ms di latenza).
    """

    def __init__(self, monitor_index: int = 1):
        self.sct = mss.mss()
        # monitor 0 = tutti i monitor, 1 = primo monitor
        self.monitor = self.sct.monitors[monitor_index]
        self.width  = self.monitor["width"]
        self.height = self.monitor["height"]

    def get_frame(self) -> np.ndarray:
        """Cattura l'intero schermo e restituisce un array BGR."""
        raw = self.sct.grab(self.monitor)
        frame = np.array(raw)
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    def get_region(self, x: int, y: int, w: int, h: int) -> np.ndarray:
        """Cattura una regione specifica dello schermo (piu' veloce del frame intero)."""
        region = {"top": y, "left": x, "width": w, "height": h}
        raw = self.sct.grab(region)
        frame = np.array(raw)
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    def get_game_region(self, margin_top: float = 0.05, margin_bottom: float = 0.15) -> np.ndarray:
        """
        Ritaglia la zona centrale del gioco escludendo HUD (barre HP, abilita').
        margin_top/bottom sono frazioni dell'altezza schermo.
        """
        top    = int(self.height * margin_top)
        bottom = int(self.height * (1.0 - margin_bottom))
        frame  = self.get_frame()
        return frame[top:bottom, :, :]

    def __del__(self):
        self.sct.close()
