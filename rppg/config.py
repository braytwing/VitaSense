"""Configuración central del sistema rPPG."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PipelineConfig:
    # --- Ventana deslizante de análisis ---
    window_s: float = 10.0        # duración de la ventana usada para cada estimación
    min_window_s: float = 6.0     # datos mínimos antes de la 1ª estimación
    step_s: float = 1.0           # cada cuánto se recalcula
    max_buffer_s: float = 30.0    # histórico máximo en memoria
    max_gap_s: float = 0.5        # cara perdida más de esto => se descarta el buffer

    # --- Procesado de señal ---
    fs_target: float | None = None              # None = auto (fps efectivo acotado a [15, 60])
    band_hz: tuple[float, float] = (0.7, 3.0)   # 42-180 lpm
    filter_order: int = 4
    algorithm: str = "pos"                      # "green" | "chrom" | "pos"
