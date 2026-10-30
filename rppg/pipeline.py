"""Pipeline rPPG: desacoplado de la cámara y de la GUI.

    frame -> ROI -> media RGB -> TraceBuffer -> (cada step_s) remuestreo
          -> algoritmo (POS/CHROM/GREEN) -> estimadores -> Snapshot

Cada estimador puede pedir su propia ventana (`window_s`): la FC usa la del pipeline
(10 s) y la respiración, p. ej., 30 s. El pipeline construye un contexto por cada
duración distinta y lo comparte entre los estimadores que la usan.

Señales auxiliares (AUX_NAMES): 'chest' (desplazamiento del tórax por flujo óptico, si se pasa
un ChestMotionTracker) y 'alar' (ancho alar nasal, experimental). Se remuestrean junto al RGB.

Uso con GUI (p. ej. CustomTkinter): llama a `process_frame(frame)` por cada frame
y pinta el Snapshot cuando no sea None.
Uso sin cámara (tests / datasets): `push_rgb(t, rgb)`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .algorithms import get_algorithm
from .config import PipelineConfig
from .dsp import resample_uniform
from .estimators import HeartRateEstimator, VitalEstimator, VitalResult, WindowContext
from .quality import QualityReport, assess
from .traces import BASE_COLS, TraceBuffer


AUX_NAMES = ("chest", "alar")


@dataclass
class Snapshot:
    t: float
    results: dict                  # nombre -> VitalResult
    quality: QualityReport
    pulse_t: np.ndarray
    pulse: np.ndarray
    fs: float

    @property
    def hr(self) -> Optional[VitalResult]:
        return self.results.get("hr")

    @property
    def rr(self) -> Optional[VitalResult]:
        return self.results.get("rr")

    @property
    def spo2(self) -> Optional[VitalResult]:
        return self.results.get("spo2")


class RPPGPipeline:
    def __init__(self, roi=None, estimators: Optional[list[VitalEstimator]] = None,
                 cfg: Optional[PipelineConfig] = None, chest=None):
        self.cfg = cfg or PipelineConfig()
        self.roi = roi
        self.chest = chest                   # ChestMotionTracker opcional
        self.estimators = estimators if estimators is not None else [HeartRateEstimator()]
        longest = max([self.cfg.window_s] + [e.window_s or 0.0 for e in self.estimators])
        self.buffer = TraceBuffer(max(self.cfg.max_buffer_s, longest + 2.0), self.cfg.max_gap_s,
                                  n_aux=len(AUX_NAMES))
        self._algo = get_algorithm(self.cfg.algorithm)
        self._next_t: Optional[float] = None
        self.last_roi = None

    def reset(self) -> None:
        self.buffer.clear()
        self._next_t = None
        if self.chest is not None:
            self.chest.reset()
        for e in self.estimators:
            e.reset()

    # -- entrada por frame (cámara real) --------------------------------------
    def process_frame(self, frame) -> Optional[Snapshot]:
        if self.roi is None:
            raise RuntimeError("process_frame requiere un ROIExtractor; usa push_rgb sin cámara")
        r = self.roi.process(frame.image)
        self.last_roi = r
        if r is None:
            if self.chest is not None:
                self.chest.break_()
            return None
        aux = {"alar": r.alar}
        if self.chest is not None:
            aux["chest"] = self.chest.update(frame.image, r.face_box)
        return self.push_rgb(frame.t, r.rgb, r.nose_xy, r.face_width, r.sat_frac, aux=aux)

    # -- entrada por muestra RGB (tests, datasets, otras cámaras) --------------
    def push_rgb(self, t: float, rgb, nose_xy=(0.0, 0.0), face_width: float = 1.0,
                 sat_frac: float = 0.0, aux: Optional[dict] = None) -> Optional[Snapshot]:
        aux_row = [(aux or {}).get(n) for n in AUX_NAMES]
        if self.buffer.append(t, rgb, nose_xy, face_width, sat_frac, aux_row):
            self._next_t = None
            for e in self.estimators:
                e.reset()
        if self.buffer.duration < self.cfg.min_window_s:
            return None
        if self._next_t is None:
            self._next_t = t
        if t < self._next_t:
            return None
        self._next_t = t + self.cfg.step_s
        return self._analyze()

    # -- construcción del contexto para una duración de ventana ---------------
    def _build_ctx(self, window_s: float, min_s: float, results: dict):
        arr = self.buffer.window(window_s)
        if arr is None:
            return None
        t = arr[:, 0]
        dur = t[-1] - t[0]
        if dur < min_s:
            return None
        fps_eff = (len(t) - 1) / dur
        if fps_eff < 8:                      # muestreo insuficiente para 3 Hz de banda
            return None
        fs = self.cfg.fs_target or float(np.clip(round(fps_eff), 15, 60))
        aux_cols = [BASE_COLS + k for k in range(len(AUX_NAMES))]
        tu, xu = resample_uniform(t, arr[:, [1, 2, 3, 4, 5] + aux_cols], fs)   # r,g,b,nose_xy,aux...
        rgbu = xu[:, :3]
        fw = max(float(arr[:, 6].mean()), 1e-6)
        nose = xu[:, 3:5] / fw
        aux = {n: xu[:, 5 + k] for k, n in enumerate(AUX_NAMES) if np.isfinite(xu[:, 5 + k]).all()}
        pulse = self._algo(rgbu, fs, self.cfg.band_hz, self.cfg.filter_order)
        motion = float(np.hypot(nose[:, 0].std(), nose[:, 1].std()))
        ctx = WindowContext(t=tu, rgb=rgbu, fs=fs, pulse=pulse, band_hz=self.cfg.band_hz,
                            motion=motion, nose=nose, aux=aux, results=results)
        return ctx, arr, fps_eff

    # -- análisis ---------------------------------------------------------------
    def _analyze(self) -> Optional[Snapshot]:
        results: dict = {}
        base = self._build_ctx(self.cfg.window_s, self.cfg.min_window_s, results)
        if base is None:
            return None
        ctx0, arr0, fps_eff = base
        cache = {self.cfg.window_s: ctx0}

        for est in self.estimators:
            w = est.window_s or self.cfg.window_s
            if w not in cache:
                built = self._build_ctx(w, est.min_window_s or self.cfg.min_window_s, results)
                cache[w] = built[0] if built else None
            ctx = cache[w]
            if ctx is None:
                need = est.min_window_s or self.cfg.min_window_s
                results[est.name] = VitalResult(est.name, None, est.unit, 0.0,
                                                {"note": "acumulando datos", "needs_s": need})
            else:
                results[est.name] = est.update(ctx)

        hr = results.get("hr")
        snr = hr.extra.get("snr_db") if hr is not None else None
        brightness = float((arr0[:, 1:4] @ np.array([0.299, 0.587, 0.114])).mean())
        q = assess(fps_eff, ctx0.motion, brightness, float(arr0[:, 7].mean()), snr,
                   hr.quality if hr is not None else 0.0)
        return Snapshot(t=float(arr0[-1, 0]), results=results, quality=q,
                        pulse_t=ctx0.t, pulse=ctx0.pulse, fs=ctx0.fs)
