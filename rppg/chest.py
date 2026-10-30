"""Movimiento del tórax/hombros por flujo óptico: fuente respiratoria 'chest'.

DISEÑO (evita un defecto frecuente): el flujo se calcula entre el frame anterior y el actual
recortando AMBOS con las MISMAS coordenadas de ROI (las del frame actual). Si en cambio se
recorta cada frame con la caja de cara de ese instante y se compara (p. ej. recorte + resize),
el temblor del detector de caras (varios px entre frames) se convierte en "movimiento de
tórax" falso, mucho mayor que la respiración (~1-3 px a 640x480).

Además: la caja de cara se suaviza mucho (EMA lenta) para que el ROI sea estable, se usan solo
píxeles con textura (la ropa lisa da flujo ruidoso) y se acumula el desplazamiento ANTES de
remuestrear (la velocidad por frame no se puede remuestrear a otra frecuencia).

Si el tórax no está en cuadro (típico con la cámara de un portátil y la cara grande),
devuelve None y `visible=False`: aleja la cámara o muestra los hombros.
"""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np


class ChestMotionTracker:
    def __init__(self, scale: float = 0.5, box_ema: float = 0.05, min_px: int = 48):
        self.scale, self.box_ema, self.min_px = scale, box_ema, min_px
        self.roi: Optional[tuple] = None     # (x0, y0, x1, y1) en px del frame original
        self.visible = False
        self.reset()

    def reset(self) -> None:
        self._cum = 0.0
        self._prev: Optional[np.ndarray] = None
        self._box: Optional[np.ndarray] = None
        self.roi, self.visible = None, False

    def break_(self) -> None:
        """Cara perdida: el siguiente flujo no debe cruzar el hueco."""
        self._prev = None

    def update(self, frame_bgr: np.ndarray, face_box) -> Optional[float]:
        """Devuelve el desplazamiento vertical ACUMULADO del tórax (px de la imagen original,
        positivo hacia abajo) o None si el tórax no es visible."""
        if face_box is None:
            self.break_()
            return None
        H, W = frame_bgr.shape[:2]
        box = np.asarray(face_box, float)
        self._box = box if self._box is None else (1 - self.box_ema) * self._box + self.box_ema * box
        bx, by, bw, bh = self._box
        x0, x1 = int(max(bx - 0.35 * bw, 0)), int(min(bx + 1.35 * bw, W))
        y0, y1 = int(max(by + 1.15 * bh, 0)), int(min(by + 2.35 * bh, H))
        self.roi = (x0, y0, x1, y1)
        if y1 - y0 < self.min_px or x1 - x0 < self.min_px:
            self.visible, self._prev = False, None
            return None
        self.visible = True

        s = self.scale
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gs = cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        if self._prev is None or self._prev.shape != gs.shape:
            self._prev = gs
            return self._cum
        sx0, sy0, sx1, sy1 = (int(round(v * s)) for v in (x0, y0, x1, y1))
        prev, cur = self._prev[sy0:sy1, sx0:sx1], gs[sy0:sy1, sx0:sx1]   # MISMAS coordenadas
        flow = cv2.calcOpticalFlowFarneback(prev, cur, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        fy = flow[..., 1]
        mag = np.hypot(cv2.Sobel(cur, cv2.CV_32F, 1, 0), cv2.Sobel(cur, cv2.CV_32F, 0, 1))
        textured = mag >= np.median(mag)
        vy = float(np.median(fy[textured] if textured.sum() >= 50 else fy))
        self._cum += vy / s
        self._prev = gs
        return self._cum
