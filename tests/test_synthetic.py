"""Ejecutar: python tests/test_synthetic.py   (o pytest)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rppg import HeartRateEstimator, PipelineConfig, RPPGPipeline  # noqa: E402
from rppg.synthetic import make_rgb_trace  # noqa: E402


def run(t, rgb, algo="pos", **cfg):
    pipe = RPPGPipeline(cfg=PipelineConfig(algorithm=algo, **cfg),
                        estimators=[HeartRateEstimator()])
    snaps = []
    for ti, ci in zip(t, rgb):
        s = pipe.push_rgb(ti, ci)
        if s is not None:
            snaps.append(s)
    return pipe, snaps


def test_hr_accuracy():
    for algo in ("pos", "chrom"):
        for hr in (55, 72, 95, 130):
            t, rgb = make_rgb_trace(hr_bpm=hr, seed=hr)
            _, snaps = run(t, rgb, algo)
            assert snaps, f"{algo}/{hr}: sin estimaciones"
            vals = [s.hr.value for s in snaps[-5:] if s.hr.value is not None]
            assert vals, f"{algo}/{hr}: sin valor de FC"
            err = abs(np.mean(vals) - hr)
            assert err < 2.0, f"{algo}/{hr}: error {err:.2f} lpm"
            print(f"  {algo:5s} FC={hr:3d} -> {np.mean(vals):6.1f} lpm (err {err:.2f})")


def test_hr_ramp():
    t, rgb = make_rgb_trace(hr_bpm=60, hr_end_bpm=90, duration_s=50, seed=3)
    _, snaps = run(t, rgb, "pos")
    last = snaps[-1].hr.value
    assert last is not None and abs(last - 87) < 6, f"seguimiento de rampa: {last}"
    print(f"  rampa 60->90: última estimación {last:.1f} lpm")


def test_gap_resets_buffer():
    t, rgb = make_rgb_trace(hr_bpm=75, duration_s=30, seed=5)
    keep = (t < 12) | (t > 14)                    # hueco de 2 s (cara perdida)
    pipe, snaps = run(t[keep], rgb[keep], "pos")
    assert pipe.buffer.duration < 20, "el buffer debió vaciarse tras el hueco"
    assert snaps[-1].hr.value is not None
    print(f"  hueco: buffer={pipe.buffer.duration:.1f}s, FC={snaps[-1].hr.value:.1f}")


def test_noise_only_is_low_quality():
    rng = np.random.default_rng(1)
    t = np.arange(0, 30, 1 / 30)
    rgb = np.array([185.0, 130.0, 115.0]) + rng.normal(0, 0.5, (len(t), 3))
    _, snaps = run(t, rgb, "pos")
    assert snaps
    qs = [s.hr.quality for s in snaps]
    assert np.mean(qs) < 0.4, f"calidad media demasiado alta con solo ruido: {np.mean(qs):.2f}"
    print(f"  solo ruido: calidad media {np.mean(qs):.2f}")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            print(name)
            fn()
    print("OK")
