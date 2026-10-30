"""Buffer de trazas temporales (RGB + movimiento + saturación + señales auxiliares)."""
from __future__ import annotations

from collections import deque
from typing import Optional, Sequence

import numpy as np

BASE_COLS = 8   # t, r, g, b, nose_x, nose_y, face_width, sat_frac


class TraceBuffer:
    """Filas: t, r, g, b, nose_x, nose_y, face_width, sat_frac, aux_0..aux_{n-1}.

    Las señales auxiliares (desplazamiento del tórax, ancho alar...) se mantienen con
    'zero-order hold' cuando una muestra no las trae; NaN hasta que aparece la primera.
    """

    def __init__(self, max_seconds: float = 30.0, max_gap_s: float = 0.5, n_aux: int = 0):
        self.max_seconds = max_seconds
        self.max_gap_s = max_gap_s
        self.n_aux = n_aux
        self._rows: deque = deque()

    def append(self, t: float, rgb, nose_xy=(0.0, 0.0), face_width: float = 1.0,
               sat_frac: float = 0.0, aux: Optional[Sequence[Optional[float]]] = None) -> bool:
        """Añade una muestra. Devuelve True si hubo un hueco largo y se vació el buffer."""
        reset = False
        if self._rows:
            last_t = self._rows[-1][0]
            if t <= last_t:              # timestamp no monótono: se ignora
                return False
            if t - last_t > self.max_gap_s:
                self._rows.clear()
                reset = True
        prev = self._rows[-1][BASE_COLS:] if self._rows else (np.nan,) * self.n_aux
        aux = aux if aux is not None else (None,) * self.n_aux
        aux_vals = tuple(float(a) if a is not None else p for a, p in zip(aux, prev))
        self._rows.append((t, float(rgb[0]), float(rgb[1]), float(rgb[2]),
                           float(nose_xy[0]), float(nose_xy[1]),
                           float(face_width), float(sat_frac)) + aux_vals)
        while self._rows and t - self._rows[0][0] > self.max_seconds:
            self._rows.popleft()
        return reset

    def window(self, seconds: float) -> Optional[np.ndarray]:
        if len(self._rows) < 2:
            return None
        arr = np.asarray(self._rows, dtype=float)
        return arr[arr[:, 0] >= arr[-1, 0] - seconds]

    @property
    def duration(self) -> float:
        return self._rows[-1][0] - self._rows[0][0] if len(self._rows) > 1 else 0.0

    def clear(self) -> None:
        self._rows.clear()
