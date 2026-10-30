"""SpO2 con webcam RGB (experimental).

Principio: la hemoglobina oxigenada y la desoxigenada absorben distinto en rojo y azul/verde,
así que la razón entre la pulsatilidad (AC/DC) de dos canales, R = (AC/DC)_rojo / (AC/DC)_ref,
varía con la saturación: SpO2 ≈ a - b·R.

Cómo se mide aquí (más robusto que un simple std):
  1. cada canal se normaliza por su DC (media) y se filtra en banda estrecha alrededor de la FC;
  2. la pulsatilidad de cada canal es su PENDIENTE de regresión sobre la señal de pulso de
     referencia (salida de POS/CHROM): lo que no está correlacionado con el pulso (ruido,
     movimiento, parpadeo) se rechaza;
  3. R_rb = pend_R / pend_B  y  R_rg = pend_R / pend_G  (se registran ambas);
  4. se toma la mediana de las últimas N razones válidas.

LÍMITES (importantes):
  - (a, b) dependen de tu cámara, iluminación y piel: SIN calibrar contra un oxímetro, el número
    no es una medida de SpO2. Por eso, por defecto el estimador solo reporta la razón R.
  - Con un sujeto sano (SpO2 96-100 %) no hay rango de referencia para calibrar bien: el ajuste
    solo captura ruido. Necesitas datos con un rango amplio (p. ej. datasets clínicos); no
    provoques desaturaciones para obtenerlos.
  - La compresión (MJPG), el auto-exposure y el balance de blancos automático dañan la AC de
    cada canal. Fíjalos, y si puedes usa `WebcamSource(fourcc=None)`.
  - Resultado orientativo/educativo; no apto para decisiones médicas.
"""
from __future__ import annotations

import json
import warnings
from collections import deque
from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np

from .dsp import bandpass
from .estimators import VitalEstimator, VitalResult, WindowContext

EPS = 1e-9
UNCALIBRATED_PLACEHOLDER = (110.0, 25.0)   # recta empírica clásica de oximetría de transmisión


@dataclass
class SpO2Calibration:
    a: float
    b: float
    pair: str = "rb"            # "rb" (rojo/azul) o "rg" (rojo/verde)
    n: int = 0
    r2: float = float("nan")
    rmse: float = float("nan")
    ref_min: float = float("nan")
    ref_max: float = float("nan")

    def apply(self, ratio: float) -> float:
        return self.a - self.b * ratio

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(asdict(self), fh, indent=2)

    @classmethod
    def load(cls, path: str) -> "SpO2Calibration":
        with open(path, encoding="utf-8") as fh:
            return cls(**json.load(fh))


def fit_calibration(ratios, refs, pair: str = "rb") -> SpO2Calibration:
    """Ajusta SpO2_ref = a - b·R por mínimos cuadrados."""
    r, y = np.asarray(ratios, float), np.asarray(refs, float)
    ok = np.isfinite(r) & np.isfinite(y)
    r, y = r[ok], y[ok]
    if len(r) < 5:
        raise ValueError("Se necesitan al menos 5 pares (razón, referencia)")
    if np.ptp(y) < 5.0:
        warnings.warn(f"El rango de SpO2 de referencia es de solo {np.ptp(y):.1f} puntos: "
                      "el ajuste no es fiable (captura ruido, no la relación real).")
    m, c = np.polyfit(r, y, 1)
    pred = c + m * r
    ss_res, ss_tot = float(((y - pred) ** 2).sum()), float(((y - y.mean()) ** 2).sum())
    return SpO2Calibration(a=float(c), b=float(-m), pair=pair, n=len(r),
                           r2=1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
                           rmse=float(np.sqrt(ss_res / len(r))),
                           ref_min=float(y.min()), ref_max=float(y.max()))


class SpO2Estimator(VitalEstimator):
    name = "spo2"
    unit = "%"

    def __init__(self, calibration=None, allow_uncalibrated: bool = False,
                 pair: str = "rb", halfband_hz: float = 0.2, history: int = 10,
                 min_ratio_quality: float = 0.25):
        """calibration: SpO2Calibration o tupla (a, b). allow_uncalibrated=True usa una recta
        genérica (solo para experimentar; el resultado se marca calibrated=False)."""
        if isinstance(calibration, tuple):
            calibration = SpO2Calibration(a=calibration[0], b=calibration[1], pair=pair)
        self.calibration = calibration
        self.allow_uncalibrated = allow_uncalibrated
        self.pair = calibration.pair if calibration is not None else pair
        self.halfband = halfband_hz
        self.min_ratio_quality = min_ratio_quality
        self._hist: deque = deque(maxlen=history)    # (R_rb, R_rg)

    def reset(self) -> None:
        self._hist.clear()

    def _fail(self, note: str, **extra) -> VitalResult:
        return VitalResult("spo2", None, "%", 0.0, {"note": note, **extra})

    def update(self, ctx: WindowContext) -> VitalResult:
        hr = ctx.results.get("hr")
        if hr is None or hr.value is None:
            return self._fail("requiere una FC válida")
        f = hr.value / 60.0
        band = (max(f - self.halfband, 0.5), f + self.halfband)
        try:
            ref = bandpass(ctx.pulse, ctx.fs, band)
            norm = ctx.rgb / (ctx.rgb.mean(axis=0, keepdims=True) + EPS) - 1.0   # AC/DC por canal
            chan = bandpass(norm, ctx.fs, band)
        except ValueError:
            return self._fail("señal insuficiente")
        pp = float(ref @ ref)
        if pp < 1e-18:
            return self._fail("pulso de referencia nulo")
        slope = (chan.T @ ref) / pp                                   # pulsatilidad R, G, B
        corr = np.array([np.corrcoef(chan[:, c], ref)[0, 1] for c in range(3)])
        q_corr = float(np.clip(min(abs(corr[0]), abs(corr[2])) / 0.6, 0.0, 1.0))
        q = float(hr.quality * q_corr)

        r_rb = slope[0] / slope[2] if abs(slope[2]) > EPS else np.nan
        r_rg = slope[0] / slope[1] if abs(slope[1]) > EPS else np.nan
        valid = np.isfinite(r_rb) and np.isfinite(r_rg) and r_rb > 0 and r_rg > 0
        if valid and q >= self.min_ratio_quality:
            self._hist.append((float(r_rb), float(r_rg)))
        extra = {"ratio_rb_raw": float(r_rb), "ratio_rg_raw": float(r_rg),
                 "corr_rgb": corr.tolist(), "n_hist": len(self._hist)}
        if not self._hist:
            return VitalResult("spo2", None, "%", q, {**extra, "note": "razón no válida"})

        med = np.median(np.array(self._hist), axis=0)
        extra.update(ratio_rb=float(med[0]), ratio_rg=float(med[1]))
        R = float(med[0] if self.pair == "rb" else med[1])

        if self.calibration is not None:
            val, calibrated = self.calibration.apply(R), True
        elif self.allow_uncalibrated:
            val, calibrated = UNCALIBRATED_PLACEHOLDER[0] - UNCALIBRATED_PLACEHOLDER[1] * R, False
        else:
            return VitalResult("spo2", None, "%", q, {**extra, "calibrated": False,
                                                      "note": "sin calibrar: solo se reporta la razón R"})
        extra["calibrated"] = calibrated
        return VitalResult("spo2", float(np.clip(val, 70.0, 100.0)), "%",
                           q if calibrated else 0.5 * q, extra)
