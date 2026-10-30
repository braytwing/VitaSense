"""Ejecutar: python tests/test_resp_spo2.py   (o pytest)."""
import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rppg import (HeartRateEstimator, PipelineConfig, RespiratoryRateEstimator,  # noqa: E402
                  RPPGPipeline, SpO2Estimator, fit_calibration)
from rppg.synthetic import TRUE_RATIO_RB, TRUE_RATIO_RG, make_rgb_trace  # noqa: E402


def run(t, rgb, nose, estimators):
    pipe = RPPGPipeline(cfg=PipelineConfig(), estimators=estimators)
    snaps = []
    for i in range(len(t)):
        s = pipe.push_rgb(t[i], rgb[i], nose_xy=tuple(nose[i]))
        if s is not None:
            snaps.append(s)
    return snaps


def last_mean(snaps, name, k=5):
    vals = [s.results[name].value for s in snaps[-k:] if s.results[name].value is not None]
    return float(np.mean(vals)) if vals else None


def test_rr_all_sources():
    for rr in (12, 18, 24):
        t, rgb, nose = make_rgb_trace(hr_bpm=72, duration_s=60, resp_bpm=rr, return_nose=True, seed=rr)
        snaps = run(t, rgb, nose, [HeartRateEstimator(), RespiratoryRateEstimator()])
        est = last_mean(snaps, "rr")
        assert est is not None and abs(est - rr) < 1.5, f"RR={rr}: estimado {est}"
        src = snaps[-1].rr.extra["sources"]
        print(f"  RR={rr:2d} -> {est:5.1f}  (q={snaps[-1].rr.quality:.2f}, "
              f"fuentes: " + ", ".join(f"{k}={v['rpm']:.1f}" for k, v in src.items()) + ")")


def test_rr_each_source_alone():
    for src in ("am", "fm", "bw", "motion"):
        t, rgb, nose = make_rgb_trace(hr_bpm=72, duration_s=60, resp_bpm=15,
                                      resp_sources=(src,), return_nose=True, seed=7)
        snaps = run(t, rgb, nose, [HeartRateEstimator(), RespiratoryRateEstimator(sources=(src,))])
        est = last_mean(snaps, "rr")
        if src == "fm":
            # La modulación de frecuencia entre latidos es la fuente más débil (la resolución
            # temporal de los picos es ~ la amplitud de la señal): debe abstenerse o acertar,
            # nunca devolver un valor confiable pero erróneo.
            assert est is None or abs(est - 15) < 3.0, f"fm sola dio {est}"
            print(f"  fuente fm     sola -> {'se abstiene (SNR bajo)' if est is None else f'{est:5.1f} resp/min'}")
            continue
        assert est is not None and abs(est - 15) < 2.0, f"fuente {src}: estimado {est}"
        print(f"  fuente {src:6s} sola -> {est:5.1f} resp/min")


def test_rr_waits_for_data_and_rejects_no_breathing():
    t, rgb, nose = make_rgb_trace(hr_bpm=72, duration_s=45, resp_bpm=None, return_nose=True, seed=2)
    snaps = run(t, rgb, nose, [HeartRateEstimator(), RespiratoryRateEstimator()])
    assert snaps[0].rr.value is None and snaps[0].rr.extra.get("note") == "acumulando datos"
    qs = [s.rr.quality for s in snaps if s.rr.extra.get("note") != "acumulando datos"]
    assert qs and np.mean(qs) < 0.5, f"calidad media sin respiración: {np.mean(qs):.2f}"
    print(f"  sin respiración: calidad media {np.mean(qs):.2f}; 1ª estimación tras acumular datos")


def test_spo2_ratio_and_calibration():
    t, rgb, nose = make_rgb_trace(hr_bpm=72, duration_s=40, illum_interference=0.0,
                                  return_nose=True, seed=11)
    snaps = run(t, rgb, nose, [HeartRateEstimator(), SpO2Estimator()])
    ex = snaps[-1].spo2.extra
    assert snaps[-1].spo2.value is None and "sin calibrar" in ex["note"]
    assert abs(ex["ratio_rb"] / TRUE_RATIO_RB - 1) < 0.10, ex["ratio_rb"]
    assert abs(ex["ratio_rg"] / TRUE_RATIO_RG - 1) < 0.10, ex["ratio_rg"]
    print(f"  R_rb={ex['ratio_rb']:.3f} (real {TRUE_RATIO_RB:.3f})  "
          f"R_rg={ex['ratio_rg']:.3f} (real {TRUE_RATIO_RG:.3f})")

    snaps = run(t, rgb, nose, [HeartRateEstimator(), SpO2Estimator(calibration=(110.0, 25.0))])
    s = snaps[-1].spo2
    assert abs(s.value - (110 - 25 * s.extra["ratio_rb"])) < 1e-6 and s.extra["calibrated"]
    print(f"  con calibración (110, 25): {s.value:.1f} %")


def test_fit_calibration():
    rng = np.random.default_rng(0)
    r = rng.uniform(0.4, 1.0, 60)
    ref = 105 - 20 * r + rng.normal(0, 0.5, 60)
    cal = fit_calibration(r, ref)
    assert abs(cal.a - 105) < 1.5 and abs(cal.b - 20) < 2.0 and cal.r2 > 0.9
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        fit_calibration(r, 97 + rng.normal(0, 0.3, 60))     # sujeto sano: sin rango
        assert any("rango" in str(x.message) for x in w)
    print(f"  ajuste: a={cal.a:.1f} b={cal.b:.1f} r2={cal.r2:.3f}; avisa si el rango es estrecho")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            print(name)
            fn()
    print("OK")
