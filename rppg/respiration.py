"""Frecuencia respiratoria (FR) por fusión espectral de varias fuentes.

La respiración deja huella en el vídeo de varias formas; cada una es una "fuente":
  am      modulación de AMPLITUD del pulso (la respiración cambia la amplitud de cada latido)
  fm      modulación de FRECUENCIA (arritmia sinusal respiratoria: el intervalo entre latidos oscila)
  bw      deriva de línea base cromática: log(G) - log(R). Restar logs cancela los cambios de
          iluminación común a los canales, que contaminarían un canal suelto
  motion  micro-movimiento vertical/horizontal de la nariz (cabeza/hombros al respirar)
  chest   desplazamiento vertical del tórax/hombros por flujo óptico (chest.py); es la fuente
          con más respaldo en la literatura de FR por vídeo, pero exige tórax en cuadro
  alar    (opt-in, EXPERIMENTAL) ancho de las alas nasales. El aleteo nasal es un signo clínico de
          dificultad respiratoria (sobre todo en niños); en adultos sanos en reposo es casi
          imperceptible y no conozco evidencia de que sirva como estimador. Entra solo si lo
          pides en `sources` y pesa únicamente si muestra un pico real (ponderación por SNR).

Si 'chest' y 'alar' están activas, `extra['alar_chest_coherence']` mide su coherencia en la
frecuencia estimada (diagnóstico de la coordinación nariz-pecho; no decide el valor).

Cada fuente se convierte en un espectro en la banda respiratoria, se pondera por su SNR y
se suman los espectros: la frecuencia donde coinciden varias fuentes gana. La calidad
refleja tanto la nitidez del pico como el acuerdo entre fuentes.

Requiere ventana larga: 6 resp/min = 0,1 Hz = un ciclo cada 10 s. Por defecto 30 s
(mínimo 20 s para dar la primera estimación).
"""
from __future__ import annotations

import numpy as np
from scipy.signal import coherence, find_peaks

from .dsp import bandpass
from .estimators import ScalarTracker, VitalEstimator, VitalResult, WindowContext

EPS = 1e-9
FS_BEATS = 4.0   # Hz, rejilla uniforme para las series latido a latido


def _grid_spectrum(x: np.ndarray, fs: float, grid_hz: np.ndarray, nfft: int = 16384):
    x = (x - x.mean()) * np.hanning(len(x))
    X = np.fft.rfft(x, nfft)
    f = np.fft.rfftfreq(nfft, 1.0 / fs)
    return np.interp(grid_hz, f, X.real ** 2 + X.imag ** 2)


def _beat_series(ctx: WindowContext, hr_bpm):
    """Devuelve {'am': (x, fs, hi), 'fm': (x, fs, hi)} o {} si hay pocos latidos."""
    ibi_exp = 60.0 / hr_bpm if hr_bpm else 0.75
    dist = max(int(ctx.fs * 0.6 * ibi_exp), 1)      # descarta picos dicrotas (<60 % del IBI)
    p = ctx.pulse
    peaks, _ = find_peaks(p, distance=dist, prominence=0.2 * p.std())
    if len(peaks) < 8:
        return {}
    # Tiempo de pico con resolución sub-muestra: a 30 fps un frame son 33 ms, comparable a la
    # modulación respiratoria del intervalo entre latidos (~±30 ms) y la taparía.
    y0, y1, y2 = p[peaks - 1], p[peaks], p[peaks + 1]
    den = y0 - 2 * y1 + y2
    delta = np.where(np.abs(den) > 1e-12, 0.5 * (y0 - y2) / np.where(den == 0, 1, den), 0.0)
    tp = ctx.t[peaks] + np.clip(delta, -1.0, 1.0) / ctx.fs
    amp = np.array([p[b] - p[a + int(np.argmin(p[a:b]))] for a, b in zip(peaks[:-1], peaks[1:])])
    ibi = np.diff(tp)
    tb = tp[1:]
    if tb[-1] - tb[0] < 10.0:
        return {}
    tu = np.arange(tb[0], tb[-1], 1.0 / FS_BEATS)
    hi = 0.45 * (len(peaks) - 1) / (tp[-1] - tp[0])  # Nyquist de la serie latido a latido
    return {"am": (np.interp(tu, tb, amp), FS_BEATS, hi),
            "fm": (np.interp(tu, tb, ibi), FS_BEATS, hi)}


class RespiratoryRateEstimator(VitalEstimator):
    name = "rr"
    unit = "resp/min"
    SOURCES = ("am", "fm", "bw", "motion", "chest")          # por defecto
    ALL_SOURCES = SOURCES + ("alar",)                         # alar: opt-in experimental

    def __init__(self, rr_range_brpm=(6.0, 30.0), window_s: float = 30.0,
                 min_window_s: float = 20.0, sources=SOURCES,
                 q_process: float = 0.1, r_base: float = 2.0, min_quality: float = 0.3,
                 max_jump_brpm: float = 6.0, hold_s: float = 20.0,
                 grid_step_hz: float = 0.004):
        self.rr_range = rr_range_brpm
        self.window_s, self.min_window_s = window_s, min_window_s
        self.sources = tuple(sources)
        self.grid_step = grid_step_hz
        self._trk = ScalarTracker(q_process, r_base, min_quality, max_jump_brpm, hold_s)

    def reset(self) -> None:
        self._trk.reset()

    def _candidates(self, ctx: WindowContext):
        c = {}
        if "bw" in self.sources:
            c["bw"] = (np.log(ctx.rgb[:, 1] + EPS) - np.log(ctx.rgb[:, 0] + EPS), ctx.fs, None)
        if "motion" in self.sources and ctx.nose is not None:
            for ax, nm in ((1, "motion_y"), (0, "motion_x")):
                if ctx.nose[:, ax].std() > 1e-7:
                    c[nm] = (ctx.nose[:, ax], ctx.fs, None)
        for nm, thr in (("chest", 1e-6), ("alar", 1e-7)):
            if nm in self.sources and nm in ctx.aux and ctx.aux[nm].std() > thr:
                c[nm] = (ctx.aux[nm], ctx.fs, None)
        if "am" in self.sources or "fm" in self.sources:
            hr = ctx.results.get("hr")
            beats = _beat_series(ctx, hr.value if hr is not None else None)
            for nm in ("am", "fm"):
                if nm in self.sources and nm in beats:
                    c[nm] = beats[nm]
        return c

    def update(self, ctx: WindowContext) -> VitalResult:
        lo, hi = self.rr_range[0] / 60.0, self.rr_range[1] / 60.0
        grid = np.arange(lo, hi + 1e-9, self.grid_step)
        T = float(ctx.t[-1] - ctx.t[0])
        hw = max(0.03, 1.0 / T)                      # semiancho del pico (Hz)

        specs = []                                   # (nombre, Pn, f_pico, snr_db, peso)
        for name, (x, fs_x, hi_x) in self._candidates(ctx).items():
            band_hi = min(hi, hi_x) if hi_x else hi
            if band_hi <= lo * 1.3:
                continue
            try:
                xf = bandpass(x, fs_x, (lo, band_hi), order=3)
            except ValueError:
                continue
            P = _grid_spectrum(xf, fs_x, grid)
            P[grid > band_hi] = 0.0
            if P.sum() <= 0:
                continue
            Pn = P / P.sum()
            fpk = float(grid[int(np.argmax(Pn))])
            near = np.abs(grid - fpk) <= hw
            snr = float(10 * np.log10((Pn[near].sum() + 1e-12) / (Pn[~near].sum() + 1e-12)))
            w = float(np.clip((snr + 3.0) / 10.0, 0.0, 1.0) ** 2)
            specs.append((name, Pn, fpk, snr, w))

        src_info = {n: {"rpm": f * 60, "snr_db": s, "w": w} for n, _, f, s, w in specs}
        W = sum(s[4] for s in specs)
        if not specs or W < 1e-6:
            return VitalResult("rr", None, self.unit, 0.0,
                               {"sources": src_info, "note": "sin fuente respiratoria fiable"})

        Pf = sum(w * Pn for _, Pn, _, _, w in specs) / W
        f_f = float(grid[int(np.argmax(Pf))])
        near = np.abs(grid - f_f) <= hw
        snr_f = float(10 * np.log10((Pf[near].sum() + 1e-12) / (Pf[~near].sum() + 1e-12)))
        agree = float(np.mean([abs(f - f_f) <= 0.03 for _, _, f, _, _ in specs]))
        quality = float(np.clip((snr_f + 1.0) / 8.0, 0.0, 1.0) * (0.4 + 0.6 * agree))

        raw = f_f * 60.0
        value, held = self._trk.update(raw, quality, float(ctx.t[-1]))
        extra = {"raw_rpm": raw, "snr_db": snr_f, "agreement": agree, "held": held,
                 "sources": src_info, "freqs_hz": grid, "power": Pf}
        if "chest" in ctx.aux and "alar" in ctx.aux and "alar" in self.sources:
            try:
                fc, cxy = coherence(bandpass(ctx.aux["chest"], ctx.fs, (lo, hi), order=3),
                                    bandpass(ctx.aux["alar"], ctx.fs, (lo, hi), order=3),
                                    fs=ctx.fs, nperseg=int(ctx.fs * 15))
                extra["alar_chest_coherence"] = float(np.interp(f_f, fc, cxy))
            except ValueError:
                pass
        return VitalResult("rr", value, self.unit, quality, extra)
