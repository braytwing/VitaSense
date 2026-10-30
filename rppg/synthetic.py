"""Trazas RGB sintéticas para probar el pipeline sin cámara.

OJO: es un modelo simplificado (pulso + respiración + interferencias + ruido).
Sirve para verificar que el código funciona, NO para estimar la precisión real.
"""
from __future__ import annotations

import numpy as np

# Modulación relativa (AC/DC) del pulso por canal R, G, B. Razones conocidas:
AC_REL = np.array([0.0020, 0.0060, 0.0035])
TRUE_RATIO_RB = AC_REL[0] / AC_REL[2]      # 0.571
TRUE_RATIO_RG = AC_REL[0] / AC_REL[1]      # 0.333


def make_rgb_trace(hr_bpm: float = 72.0, duration_s: float = 40.0, fps: float = 30.0,
                   hr_end_bpm: float | None = None, noise_std: float = 0.15,
                   jitter_s: float = 0.004, illum_interference: float = 0.003,
                   resp_bpm: float | None = None,
                   resp_sources=("am", "fm", "bw", "motion"),
                   return_nose: bool = False, seed: int = 0):
    """Devuelve (t, rgb) o (t, rgb, nose) con rgb de forma (N, 3) en orden R, G, B.

    - Timestamps con jitter (como una webcam real).
    - Pulso con 2º armónico; modulación relativa mayor en G que en R/B (AC_REL).
    - Interferencia de iluminación común a los 3 canales (CHROM/POS la cancelan, GREEN no)
      y deriva lenta de iluminación (0,13 Hz, dentro de la banda respiratoria).
    - `resp_bpm`: añade respiración por las fuentes elegidas:
        am     amplitud del pulso modulada ±25 %
        fm     FC modulada ±4 % (arritmia sinusal respiratoria)
        bw     deriva de línea base con amplitud DISTINTA por canal (si fuera igual en los
               3 canales, un cociente de canales la cancelaría como si fuera iluminación)
        motion nariz oscilando en vertical (0.003 del ancho de cara) + ruido
    - `hr_end_bpm` permite una rampa de FC.
    """
    rng = np.random.default_rng(seed)
    n = int(duration_s * fps)
    t = np.arange(n) / fps + rng.normal(0.0, jitter_s, n)
    t = np.maximum.accumulate(t)
    t += np.arange(n) * 1e-9                      # estrictamente creciente

    f = np.linspace(hr_bpm, hr_end_bpm if hr_end_bpm is not None else hr_bpm, n) / 60.0
    resp = None
    if resp_bpm is not None:
        fr = resp_bpm / 60.0
        resp = np.sin(2 * np.pi * fr * t + 0.5)
        if "fm" in resp_sources:
            f = f * (1.0 + 0.04 * resp)
    phase = 2 * np.pi * np.cumsum(f) / fps
    pulse = np.sin(phase) + 0.25 * np.sin(2 * phase + 0.8)
    if resp is not None and "am" in resp_sources:
        pulse = pulse * (1.0 + 0.25 * resp)

    base = np.array([185.0, 130.0, 115.0])
    illum = (1.0 + 0.02 * np.sin(2 * np.pi * 0.13 * t)
             + illum_interference * np.sin(2 * np.pi * 1.45 * t + 0.5))
    rgb = base * (1.0 + AC_REL * pulse[:, None]) * illum[:, None]
    if resp is not None and "bw" in resp_sources:
        rgb = rgb * (1.0 + np.array([0.001, 0.004, 0.0015]) * resp[:, None])
    rgb += rng.normal(0.0, noise_std, rgb.shape)

    if not return_nose:
        return t, rgb
    nose = np.zeros((n, 2))
    nose += rng.normal(0.0, 0.0005, nose.shape)
    if resp is not None and "motion" in resp_sources:
        nose[:, 1] += 0.003 * np.sin(2 * np.pi * (resp_bpm / 60.0) * t + 2.0)
    return t, rgb, nose
