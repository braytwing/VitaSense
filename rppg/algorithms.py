"""Algoritmos de extracción de la señal de pulso a partir de trazas RGB.

Todos: f(rgb (N,3 en orden R,G,B), fs, band, order) -> pulso filtrado (N,)

- green: canal verde (baseline; sensible a luz y movimiento)
- chrom: de Haan & Jeanne, 2013 (proyección cromática)
- pos:   Wang et al., 2017 (Plane-Orthogonal-to-Skin), ventana corta con overlap-add
"""
from __future__ import annotations

import numpy as np

from .dsp import bandpass

EPS = 1e-9


def green(rgb, fs, band, order=4):
    g = rgb[:, 1]
    return bandpass(g / (g.mean() + EPS) - 1.0, fs, band, order)


def chrom(rgb, fs, band, order=4):
    n = rgb / (rgb.mean(axis=0, keepdims=True) + EPS)
    r, g, b = n[:, 0], n[:, 1], n[:, 2]
    xf = bandpass(3 * r - 2 * g, fs, band, order)
    yf = bandpass(1.5 * r + g - 1.5 * b, fs, band, order)
    alpha = xf.std() / (yf.std() + EPS)
    return xf - alpha * yf


_P = np.array([[0.0, 1.0, -1.0], [-2.0, 1.0, 1.0]])


def pos(rgb, fs, band, order=4, win_s: float = 1.6):
    T = rgb.shape[0]
    L = int(round(win_s * fs))
    if T < L:
        raise ValueError("Señal más corta que la ventana de POS")
    H = np.zeros(T)
    for n in range(L - 1, T):
        m = n - L + 1
        c = rgb[m:n + 1].T                                  # (3, L)
        cn = c / (c.mean(axis=1, keepdims=True) + EPS)
        s = _P @ cn
        h = s[0] + (s[0].std() / (s[1].std() + EPS)) * s[1]
        H[m:n + 1] += h - h.mean()
    return bandpass(H, fs, band, order)


ALGORITHMS = {"green": green, "chrom": chrom, "pos": pos}


def get_algorithm(name: str):
    try:
        return ALGORITHMS[name.lower()]
    except KeyError:
        raise ValueError(f"Algoritmo '{name}' desconocido. Opciones: {list(ALGORITHMS)}")
