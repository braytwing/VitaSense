"""Evaluación de calidad de la medición (umbrales heurísticos, ajústalos a tu cámara)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class QualityReport:
    fps: float
    motion: float                 # desplazamiento relativo al ancho de cara
    brightness: float             # luma media del ROI (0-255)
    saturation: float             # fracción de píxeles saturados
    snr_db: Optional[float]
    score: float                  # 0..1
    flags: list = field(default_factory=list)


def assess(fps: float, motion: float, brightness: float, saturation: float,
           snr_db: Optional[float], hr_quality: float) -> QualityReport:
    flags = []
    if fps < 15:
        flags.append("FPS_BAJO")
    if brightness < 60:
        flags.append("POCA_LUZ")
    if brightness > 235 or saturation > 0.05:
        flags.append("SOBREEXPUESTO")
    if motion > 0.02:
        flags.append("MOVIMIENTO")
    if snr_db is not None and snr_db < 0:
        flags.append("SNR_BAJO")
    score = hr_quality * (1.0 - 0.5 * float(np.clip(motion / 0.04, 0, 1)))
    if "POCA_LUZ" in flags or "SOBREEXPUESTO" in flags:
        score *= 0.6
    return QualityReport(fps, motion, brightness, saturation, snr_db, float(score), flags)
