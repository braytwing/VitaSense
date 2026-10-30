"""Ejecutar: python tests/test_chest.py   (o pytest). Vídeo sintético: tórax que sube y baja."""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rppg import (ChestMotionTracker, HeartRateEstimator, PipelineConfig,  # noqa: E402
                  RespiratoryRateEstimator, RPPGPipeline)
from rppg.synthetic import make_rgb_trace  # noqa: E402

RR, AMP = 15.0, 1.5      # resp/min y amplitud del tórax en px (a 640x480 es realista)


def make_base(seed=0):
    rng = np.random.default_rng(seed)
    b = cv2.GaussianBlur(rng.random((480, 640)).astype(np.float32), (0, 0), 3)
    return ((b - b.min()) / (b.max() - b.min()) * 200 + 30).astype(np.float32)


def frame_at(base, dy):
    warped = cv2.warpAffine(base, np.float32([[1, 0, 0], [0, 1, dy]]), (640, 480),
                            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    img = base.copy()
    img[260:] = warped[260:]                 # solo el "tórax" (debajo de la cara) se mueve
    return cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2BGR)


def naive_cum(frames, boxes):
    """Recorte relativo a la caja de CADA frame + resize + flujo (el defecto a evitar)."""
    prev, cum, out = None, 0.0, []
    for f, (x, y, w, h) in zip(frames, boxes):
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        c = cv2.resize(g[int(y + 1.1 * h):min(480, int(y + 2.2 * h)),
                       max(0, int(x - .1 * w)):min(640, int(x + 1.1 * w))], (100, 80))
        if prev is not None:
            cum += float(np.median(cv2.calcOpticalFlowFarneback(prev, c, None, .5, 3, 15, 3, 5, 1.2, 0)[..., 1]))
        prev = c
        out.append(cum)
    return np.array(out)


def estimate_rr(cum, t, rgb, sources=("chest",), alar=None):
    pipe = RPPGPipeline(estimators=[HeartRateEstimator(), RespiratoryRateEstimator(sources=sources)])
    snaps = []
    for i in range(len(t)):
        aux = {"chest": cum[i]}
        if alar is not None:
            aux["alar"] = alar[i]
        s = pipe.push_rgb(t[i], rgb[i], aux=aux)
        if s is not None:
            snaps.append(s)
    return snaps[-1].rr


def test_chest_invariant_to_face_box_jitter():
    t, rgb = make_rgb_trace(hr_bpm=72, duration_s=50, seed=1)
    base = make_base()
    frames = [frame_at(base, AMP * np.sin(2 * np.pi * RR / 60 * ti)) for ti in t]
    snr_mine = {}
    for jit in (3.0, 8.0):
        rng = np.random.default_rng(5)
        boxes = [(240 + rng.uniform(-jit, jit), 60 + rng.uniform(-jit, jit), 160, 190) for _ in t]
        tr = ChestMotionTracker()
        mine = np.array([tr.update(f, b) for f, b in zip(frames, boxes)], float)
        assert tr.visible
        r_mine = estimate_rr(mine, t, rgb)
        r_naive = estimate_rr(naive_cum(frames, boxes), t, rgb)
        assert r_mine.value is not None and abs(r_mine.value - RR) < 1.0, r_mine.value
        snr_mine[jit] = r_mine.extra["snr_db"]
        nv = "None" if r_naive.value is None else f"{r_naive.value:.1f}"
        print(f"  caja ±{jit:.0f}px: anclado {r_mine.value:.1f} resp/min (SNR {r_mine.extra['snr_db']:.1f} dB)"
              f" | recorte relativo {nv} (SNR {r_naive.extra['snr_db']:.1f} dB, q={r_naive.quality:.2f})")
    assert abs(snr_mine[3.0] - snr_mine[8.0]) < 1.5, "el ROI anclado no debería depender del temblor"


def test_chest_not_visible_returns_none():
    base = make_base()
    tr = ChestMotionTracker()
    img = frame_at(base, 0.0)
    assert tr.update(img, (240, 300, 160, 190)) is None and not tr.visible   # cara baja: sin tórax
    assert tr.update(img, None) is None                                       # cara perdida
    print("  tórax fuera de cuadro -> None (se avisa en la UI)")


def test_alar_is_gated_by_snr_and_coherence_reported():
    rng = np.random.default_rng(3)
    t, rgb = make_rgb_trace(hr_bpm=72, duration_s=60, seed=2)
    chest = AMP * np.sin(2 * np.pi * RR / 60 * t) + rng.normal(0, 0.1, len(t))

    # a) alar con respiración: coherente con el pecho
    alar_ok = 0.30 + 0.002 * np.sin(2 * np.pi * RR / 60 * t + 0.4) + rng.normal(0, 0.0003, len(t))
    r = estimate_rr(chest, t, rgb, ("chest", "alar"), alar=alar_ok)
    assert abs(r.value - RR) < 1.0 and r.extra["alar_chest_coherence"] > 0.8, r.extra
    print(f"  alar con respiración: RR={r.value:.1f}, coherencia pecho-alar {r.extra['alar_chest_coherence']:.2f}")

    # b) alar = solo ruido (caso adulto sano en reposo): no debe arrastrar la estimación
    alar_noise = 0.30 + rng.normal(0, 0.003, len(t))
    r = estimate_rr(chest, t, rgb, ("chest", "alar"), alar=alar_noise)
    w = r.extra["sources"]["alar"]["w"]
    assert abs(r.value - RR) < 1.0 and w < 0.2, (r.value, w)
    print(f"  alar solo ruido: RR={r.value:.1f}, peso de la fuente {w:.2f} (coherencia {r.extra['alar_chest_coherence']:.2f})")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            print(name)
            fn()
    print("OK")
