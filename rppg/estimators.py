"""Base de estimadores de signos vitales (plug-ins del pipeline) + FC + esqueleto de PA.

Para añadir una variable nueva: hereda de VitalEstimator, define `name`, `unit` e
implementa `update(ctx) -> VitalResult`. Cada estimador puede pedir su propia ventana
(`window_s`, `min_window_s`): la FC usa 10 s, la respiración ~30 s. Los estimadores
corren en orden y pueden leer lo calculado antes en `ctx.results` (p. ej. la FC).

Otros módulos:  respiration.py (FR)   bp.py (PA, investigación)   spo2.py (SpO2, experimental)
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class WindowContext:
    t: np.ndarray                       # tiempos de la rejilla uniforme (s)
    rgb: np.ndarray                     # (N, 3) RGB remuestreado
    fs: float
    pulse: np.ndarray                   # señal de pulso filtrada (algoritmo elegido)
    band_hz: tuple[float, float]
    motion: float = 0.0                 # desplazamiento relativo al ancho de cara
    nose: Optional[np.ndarray] = None   # (N, 2) posición de la nariz / ancho de cara
    aux: dict = field(default_factory=dict)       # señales auxiliares remuestreadas: 'chest', 'alar'
    results: dict = field(default_factory=dict)   # resultados de estimadores previos


@dataclass
class VitalResult:
    name: str
    value: Optional[float]
    unit: str
    quality: float                      # 0..1 (heurístico)
    extra: dict = field(default_factory=dict)


class VitalEstimator(ABC):
    name: str = "vital"
    unit: str = ""
    window_s: Optional[float] = None        # None = usa PipelineConfig.window_s
    min_window_s: Optional[float] = None    # None = usa PipelineConfig.min_window_s

    @abstractmethod
    def update(self, ctx: WindowContext) -> VitalResult: ...

    def reset(self) -> None:
        pass


# ----------------------------------------------------------------------------
# Seguimiento temporal (Kalman escalar) reutilizable por FC, FR, ...
# ----------------------------------------------------------------------------
class ScalarTracker:
    """Kalman escalar con varianza de medida R = r_base / calidad.

    q_process: varianza de proceso por segundo; r_base: varianza de medida con calidad 1.
    Si la medida es mala o es un salto improbable, mantiene el último valor hasta
    `hold_s` segundos y luego se reinicia.
    """

    def __init__(self, q_process: float, r_base: float, min_quality: float = 0.3,
                 max_jump: float = 25.0, hold_s: float = 8.0):
        self.q_process, self.r_base = q_process, r_base
        self.min_quality, self.max_jump, self.hold_s = min_quality, max_jump, hold_s
        self.reset()

    def reset(self) -> None:
        self._x: Optional[float] = None
        self._p = 0.0
        self._t_last = 0.0
        self._t_good = 0.0

    def update(self, z: float, q: float, t: float):
        """Devuelve (valor, held). valor=None si aún no hay pista fiable."""
        if self._x is None:
            if q < self.min_quality:
                return None, False
            self._x, self._p, self._t_last, self._t_good = z, self.r_base, t, t
            return self._x, False
        dt = max(t - self._t_last, 0.0)
        self._t_last = t
        self._p += self.q_process * dt
        outlier = abs(z - self._x) > self.max_jump and q < 0.7
        if q < self.min_quality or outlier:
            if t - self._t_good > self.hold_s:
                self.reset()
                return None, False
            return self._x, True
        r = self.r_base / max(q, 0.05)
        k = self._p / (self._p + r)
        self._x += k * (z - self._x)
        self._p *= (1.0 - k)
        self._t_good = t
        return self._x, False


# ----------------------------------------------------------------------------
# Frecuencia cardíaca
# ----------------------------------------------------------------------------
def _spectrum(x: np.ndarray, fs: float, nfft_min: int):
    x = (x - x.mean()) * np.hanning(len(x))
    nfft = max(nfft_min, 1 << int(np.ceil(np.log2(len(x)))))
    X = np.fft.rfft(x, nfft)
    return np.fft.rfftfreq(nfft, 1.0 / fs), X.real ** 2 + X.imag ** 2


def _parabolic_peak(f: np.ndarray, p: np.ndarray, i: int) -> float:
    if i <= 0 or i >= len(p) - 1:
        return float(f[i])
    y0, y1, y2 = np.log(p[i - 1:i + 2] + 1e-20)
    den = y0 - 2 * y1 + y2
    if abs(den) < 1e-12:
        return float(f[i])
    delta = float(np.clip(0.5 * (y0 - y2) / den, -1.0, 1.0))
    return float(f[i] + delta * (f[1] - f[0]))


def _snr_db(f, p, f0, halfwidth):
    """SNR espectral (de Haan): energía alrededor de f0 y 2*f0 vs. el resto de la banda."""
    sig = (np.abs(f - f0) <= halfwidth) | (np.abs(f - 2 * f0) <= halfwidth)
    return float(10 * np.log10((p[sig].sum() + 1e-20) / (p[~sig].sum() + 1e-20)))


class HeartRateEstimator(VitalEstimator):
    name = "hr"
    unit = "lpm"

    def __init__(self, hr_range_bpm=(42.0, 180.0), nfft: int = 4096,
                 q_process: float = 2.0, r_base: float = 4.0,
                 min_quality: float = 0.3, max_jump_bpm: float = 25.0,
                 hold_s: float = 8.0):
        self.hr_range = hr_range_bpm
        self.nfft = nfft
        self._trk = ScalarTracker(q_process, r_base, min_quality, max_jump_bpm, hold_s)

    def reset(self) -> None:
        self._trk.reset()

    def update(self, ctx: WindowContext) -> VitalResult:
        f, p = _spectrum(ctx.pulse, ctx.fs, self.nfft)
        band = (f >= self.hr_range[0] / 60.0) & (f <= self.hr_range[1] / 60.0)
        fb, pb = f[band], p[band]
        i = int(np.argmax(pb))
        f0 = _parabolic_peak(fb, pb, i)
        T = max(float(ctx.t[-1] - ctx.t[0]), 1e-3)
        snr = _snr_db(fb, pb, f0, halfwidth=max(0.1, 1.0 / T))
        quality = float(np.clip((snr + 2.0) / 10.0, 0.0, 1.0))   # -2 dB -> 0, 8 dB -> 1
        raw = f0 * 60.0
        value, held = self._trk.update(raw, quality, float(ctx.t[-1]))
        return VitalResult("hr", value, "lpm", quality,
                           {"raw_bpm": raw, "snr_db": snr, "held": held,
                            "freqs_hz": fb, "power": pb})
