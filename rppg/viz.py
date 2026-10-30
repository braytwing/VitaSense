"""Paneles de visualización (solo OpenCV, sin matplotlib).

  1. Onda de pulso rPPG de los últimos segundos, con cada latido detectado marcado.
  2. Evolución de la FC durante la medición: estimación suavizada (Kalman) y estimaciones
     crudas por ventana (puntos grises) para ver cuánto "ayuda" el seguimiento.
  3. Evolución de la FR (si hay).

Los textos son ASCII porque las fuentes Hershey de OpenCV no dibujan acentos.
"""
from __future__ import annotations

from collections import deque
from typing import Optional

import cv2
import numpy as np
from scipy.signal import find_peaks

BG, GRID, TXT, DIM = (28, 28, 28), (60, 60, 60), (225, 225, 225), (150, 150, 150)
GREEN, ORANGE, YELLOW, RED = (110, 230, 110), (70, 170, 240), (0, 220, 255), (80, 80, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _text(img, s, org, scale=0.5, color=TXT, thick=1):
    cv2.putText(img, s, org, FONT, scale, color, thick, cv2.LINE_AA)


def _mapper(rect, xlim, ylim):
    x0, y0, x1, y1 = rect
    sx = (x1 - x0) / max(xlim[1] - xlim[0], 1e-9)
    sy = (y1 - y0) / max(ylim[1] - ylim[0], 1e-9)
    return lambda x, y: (int(x0 + (x - xlim[0]) * sx), int(y1 - (y - ylim[0]) * sy))


def _axes(img, rect, title, xlim, ylim, ylabel_fmt="{:.0f}", nticks=4):
    x0, y0, x1, y1 = rect
    _text(img, title, (x0, y0 - 8), 0.5)
    cv2.rectangle(img, (x0, y0), (x1, y1), GRID, 1)
    for i in range(nticks + 1):
        v = ylim[0] + (ylim[1] - ylim[0]) * i / nticks
        y = int(y1 - (y1 - y0) * i / nticks)
        if 0 < i < nticks:
            cv2.line(img, (x0, y), (x1, y), GRID, 1)
        _text(img, ylabel_fmt.format(v), (x0 - 46, y + 4), 0.38, DIM)
    for i in range(5):
        t = xlim[0] + (xlim[1] - xlim[0]) * i / 4
        _text(img, f"{t:.0f}s", (x0 + int((x1 - x0) * i / 4) - 8, y1 + 14), 0.38, DIM)


def _polyline_segments(img, xs, ys, mp, color, thick=2):
    """Dibuja tramos contiguos (None rompe la línea)."""
    seg = []
    for x, y in zip(xs, ys):
        if y is None or not np.isfinite(y) or not np.isfinite(x):
            if len(seg) > 1:
                cv2.polylines(img, [np.array(seg, np.int32)], False, color, thick, cv2.LINE_AA)
            seg = []
        else:
            point = mp(x, y)
            if all(-1000000 <= v <= 1000000 for v in point):
                seg.append(point)
            else:
                if len(seg) > 1:
                    cv2.polylines(img, [np.array(seg, np.int32)], False, color, thick, cv2.LINE_AA)
                seg = []
    if len(seg) > 1:
        cv2.polylines(img, [np.array(seg, np.int32)], False, color, thick, cv2.LINE_AA)


def _span(vals, min_span, pad=0.15):
    lo, hi = float(np.min(vals)), float(np.max(vals))
    c, span = (lo + hi) / 2, max(hi - lo, min_span)
    return c - span * (0.5 + pad), c + span * (0.5 + pad)


class Dashboard:
    def __init__(self, width: int = 900, history_s: float = 120.0, wave_s: float = 10.0,
                 show_rr: bool = True, show_quality: bool = False):
        self.w, self.history_s, self.wave_s, self.show_rr = width, history_s, wave_s, show_rr
        self.show_quality = show_quality
        self.hist: deque = deque()          # (t, hr, hr_raw, quality, rr)
        self._t0: Optional[float] = None

    def reset(self) -> None:
        self.hist.clear()
        self._t0 = None

    def update(self, snap) -> None:
        if snap is None or snap.hr is None:
            return
        if self._t0 is None:
            self._t0 = snap.t
        rr = snap.rr.value if snap.rr is not None else None
        self.hist.append((snap.t - self._t0, snap.hr.value, snap.hr.extra.get("raw_bpm"),
                          snap.hr.quality, rr, snap.quality.score, snap.rr.quality if snap.rr else None))
        while self.hist and self.hist[-1][0] - self.hist[0][0] > self.history_s:
            self.hist.popleft()

    # ------------------------------------------------------------------
    def _wave_panel(self, img, rect, snap):
        x0, y0, x1, y1 = rect
        if snap is None or len(snap.pulse) < 10:
            _axes(img, rect, "Onda de pulso rPPG (calibrando...)", (0, self.wave_s), (-1, 1), "{:.1f}")
            return
        n = min(len(snap.pulse), int(self.wave_s * snap.fs))
        p, t = snap.pulse[-n:], snap.pulse_t[-n:] - snap.pulse_t[-n]
        z = (p - np.median(p)) / (np.percentile(np.abs(p - np.median(p)), 95) * 2.5 + 1e-12)
        z = np.clip(z, -1, 1)
        hr = snap.hr.value if snap.hr is not None and snap.hr.value else None
        dist = max(int(snap.fs * 0.6 * (60.0 / hr if hr else 0.75)), 1)
        peaks, _ = find_peaks(z, distance=dist, prominence=0.15)
        ibi = np.diff(t[peaks]) if len(peaks) > 1 else np.array([])
        sub = f"{len(peaks)} latidos"
        if len(ibi):
            sub += f" | IBI {ibi.mean() * 1000:.0f} ms | SDNN {ibi.std() * 1000:.0f} ms"
        _axes(img, rect, f"Onda de pulso rPPG (unidades relativas, no es ECG)  -  {sub}",
              (float(t[0]), float(t[-1])), (-1, 1), "{:.1f}")
        mp = _mapper(rect, (float(t[0]), float(t[-1])), (-1, 1))
        _polyline_segments(img, t, z, mp, GREEN, 1)
        for k in peaks:
            cv2.circle(img, mp(t[k], z[k]), 4, RED, -1, cv2.LINE_AA)

    def _series_panel(self, img, rect, title, idx, color, min_span, unit, raw_idx=None):
        h = list(self.hist)
        xlim = (max(0.0, h[-1][0] - self.history_s), max(h[-1][0], 30.0)) if h else (0, 30)
        vals = [r[idx] for r in h if r[idx] is not None]
        if raw_idx is not None:
            vals += [r[raw_idx] for r in h if r[raw_idx] is not None]
        if not vals:
            _axes(img, rect, f"{title} (sin datos todavia)", xlim, (0, 1))
            return
        ylim = _span(vals, min_span)
        stats = f"media {np.mean([r[idx] for r in h if r[idx] is not None]):.1f} | min {min(vals):.0f} | max {max(vals):.0f}"
        _axes(img, rect, f"{title}  -  {stats}", xlim, ylim)
        mp = _mapper(rect, xlim, ylim)
        if raw_idx is not None:
            for r in h:
                if r[raw_idx] is not None and ylim[0] <= r[raw_idx] <= ylim[1]:
                    cv2.circle(img, mp(r[0], r[raw_idx]), 2, (120, 120, 120), -1, cv2.LINE_AA)
        _polyline_segments(img, [r[0] for r in h], [r[idx] for r in h], mp, color, 2)
        last = next((r for r in reversed(h) if r[idx] is not None), None)
        if last is not None:
            label = f"{last[idx]:.1f} {unit}"
            (tw, _), _b = cv2.getTextSize(label, FONT, 0.65, 2)
            _text(img, label, (rect[2] - tw - 8, rect[1] + 22), 0.65, color, 2)

    def render(self, snap=None) -> np.ndarray:
        panels = (3 if self.show_rr else 2) + int(self.show_quality)
        ph, gap, top = 150, 42, 30
        H = top + panels * (ph + gap) + 6
        img = np.full((H, self.w, 3), BG, np.uint8)
        _text(img, "SIGNOS VITALES - SOLO INVESTIGACION", (14, 20), 0.55, YELLOW)
        x0, x1 = 64, self.w - 20
        rects = [(x0, top + gap + i * (ph + gap) - 10, x1, top + gap + i * (ph + gap) - 10 + ph) for i in range(panels)]
        self._wave_panel(img, rects[0], snap)
        self._series_panel(img, rects[1], "Evolucion de la FC (lpm)  [puntos: estimacion por ventana]",
                           1, GREEN, 10.0, "lpm", raw_idx=2)
        if self.show_rr:
            self._series_panel(img, rects[2], "Evolucion de la FR (resp/min)", 4, ORANGE, 6.0, "resp/min")
        if self.show_quality:
            rect = rects[-1]
            h = list(self.hist)
            xlim = (max(0.0, h[-1][0] - self.history_s), max(h[-1][0], 30.0)) if h else (0, 30)
            _axes(img, rect, "Calidad heuristica (0 a 1)  |  verde: FC  naranja: FR  gris: global", xlim, (0, 1), "{:.2f}")
            mp = _mapper(rect, xlim, (0, 1))
            if h:
                _polyline_segments(img, [r[0] for r in h], [r[3] for r in h], mp, GREEN)
                _polyline_segments(img, [r[0] for r in h], [r[6] for r in h], mp, DIM)
                if self.show_rr:
                    _polyline_segments(img, [r[0] for r in h], [r[7] for r in h], mp, ORANGE)
        return img
