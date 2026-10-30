"""Utilidades de procesado de señal."""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt


def resample_uniform(t: np.ndarray, x: np.ndarray, fs: float):
    """Remuestrea a rejilla uniforme (las webcams entregan timestamps irregulares)."""
    t = np.asarray(t, float)
    x = np.asarray(x, float)
    keep = np.concatenate(([True], np.diff(t) > 1e-6))
    t, x = t[keep], x[keep]
    n = int(np.floor((t[-1] - t[0]) * fs)) + 1
    tu = t[0] + np.arange(n) / fs
    if x.ndim == 1:
        return tu, np.interp(tu, t, x)
    return tu, np.column_stack([np.interp(tu, t, x[:, c]) for c in range(x.shape[1])])


def bandpass(x: np.ndarray, fs: float, band: tuple[float, float], order: int = 4):
    """Butterworth pasa-banda de fase cero (sosfiltfilt) a lo largo del eje 0."""
    if x.shape[0] <= 3 * (2 * order + 1):
        raise ValueError("Señal demasiado corta para filtrar")
    nyq = fs / 2.0
    lo = band[0] / nyq
    hi = min(band[1], 0.95 * nyq) / nyq
    if not 0 < lo < hi < 1:
        raise ValueError(f"Banda {band} inválida para fs={fs}")
    sos = butter(order, [lo, hi], btype="band", output="sos")
    return sosfiltfilt(sos, x, axis=0)
