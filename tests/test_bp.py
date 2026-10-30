"""Ejecutar: python tests/test_bp.py   (o pytest). Prueban la maquinaria, NO la validez fisiológica."""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rppg import (BloodPressureEstimator, BPCalibration, HeartRateEstimator,  # noqa: E402
                  RidgeFeatureModel, RPPGPipeline, TorchBPModel, cross_validate_sessions,
                  load_session, save_session)
from rppg.bp import BPModel, FEATURE_KEYS, make_bp_window, pulse_features  # noqa: E402
from rppg.synthetic import make_rgb_trace  # noqa: E402


class ConstModel(BPModel):
    def predict(self, feats, window):
        return 118.0, 76.0


def run(estimators, t, rgb):
    pipe = RPPGPipeline(estimators=estimators)
    snaps = [s for s in (pipe.push_rgb(t[i], rgb[i]) for i in range(len(t))) if s is not None]
    return snaps


def test_features_and_window():
    t, rgb = make_rgb_trace(hr_bpm=72, duration_s=30, seed=1)
    snap = run([HeartRateEstimator()], t, rgb)[-1]
    f = pulse_features(snap.pulse, snap.fs)
    assert f is not None and set(f) == set(FEATURE_KEYS)
    assert abs(f["hr_bpm"] - 72) < 3 and 0.1 < f["t_peak_frac"] < 0.8, f
    w = make_bp_window(snap.pulse, snap.fs)
    assert w.shape == (750,) and abs(w.min()) < 1e-6 and abs(w.max() - 1) < 1e-6
    print(f"  rasgos: FC {f['hr_bpm']:.1f}, t_pico {f['t_peak_frac']:.2f}, ancho50 {f['width50_frac']:.2f}; ventana (750,) en [0,1]")


def test_estimator_gating():
    t, rgb = make_rgb_trace(hr_bpm=72, duration_s=30, seed=2)
    # sin modelo
    s = run([HeartRateEstimator(), BloodPressureEstimator()], t, rgb)[-1].results["bp"]
    assert s.value is None and "sin modelo" in s.extra["note"]
    # modelo pero sin calibrar: no da número, solo la predicción cruda en extra
    s = run([HeartRateEstimator(), BloodPressureEstimator(model=ConstModel())], t, rgb)[-1].results["bp"]
    assert s.value is None and s.extra["raw_sbp"] == 118.0 and "sin calibrar" in s.extra["note"]
    # modelo + calibración
    cal = BPCalibration(sbp_offset=4.0, dbp_offset=-2.0)
    s = run([HeartRateEstimator(), BloodPressureEstimator(model=ConstModel(), calibration=cal)], t, rgb)[-1].results["bp"]
    assert abs(s.value - 122.0) < 1e-6 and s.extra["calibrated"] and abs(s.extra["dbp"] - 74.0) < 1e-6
    assert s.quality <= 0.5                                            # tope: evidencia limitada
    # sin FC fiable (solo ruido) no hay PA
    rng = np.random.default_rng(0)
    tt = np.arange(0, 30, 1 / 30)
    noise = np.array([185.0, 130.0, 115.0]) + rng.normal(0, 0.5, (len(tt), 3))
    s = run([HeartRateEstimator(), BloodPressureEstimator(model=ConstModel(), calibration=cal)], tt, noise)[-1].results["bp"]
    assert s.value is None
    print("  sin modelo -> None | sin calibrar -> None (raw en extra) | calibrado -> 122/74 | sin FC fiable -> None")


def test_calibration_fit_and_roundtrips():
    preds = np.array([[110.0, 70.0], [112.0, 72.0]])
    cuff = np.array([[120.0, 80.0], [124.0, 78.0]])
    cal = BPCalibration.fit(preds, cuff)
    assert abs(cal.sbp_offset - 11.0) < 1e-9 and abs(cal.dbp_offset - 8.0) < 1e-9   # media de cuff - pred
    with tempfile.TemporaryDirectory() as d:
        cal.save(os.path.join(d, "c.json"))
        assert BPCalibration.load(os.path.join(d, "c.json")) == cal
        rng = np.random.default_rng(0)
        X = rng.normal(size=(50, len(FEATURE_KEYS)))
        y = np.c_[120 + 5 * X[:, 0], 80 + 3 * X[:, 1]]
        m = RidgeFeatureModel().fit(X, y, alpha=0.1)
        m.save(os.path.join(d, "m.npz"))
        m2 = RidgeFeatureModel.load(os.path.join(d, "m.npz"))
        assert np.allclose(m.predict_matrix(X), m2.predict_matrix(X))
        # sesión recolectada -> npz -> cargada
        t, rgb = make_rgb_trace(hr_bpm=72, duration_s=30, seed=4)
        est = BloodPressureEstimator(collect=True)
        run([HeartRateEstimator(), est], t, rgb)
        assert est.records
        save_session(os.path.join(d, "s.npz"), est.records, ref=(118, 76), subject="test")
        s = load_session(os.path.join(d, "s.npz"))
        assert s["feats"].shape == (len(est.records), len(FEATURE_KEYS)) and s["windows"].shape[1] == 750
        assert tuple(s["ref"]) == (118.0, 76.0)
    print("  calibración, ridge y sesión: guardar/cargar coherentes")


def _fake_sessions(n, informative, seed):
    rng = np.random.default_rng(seed)
    out = []
    lat = rng.normal(size=n)                                    # PA latente por sesión
    for i in range(n):
        base = rng.normal(0, 1, len(FEATURE_KEYS)) * 0.2
        feats = np.tile(base, (12, 1)) + rng.normal(0, 0.05, (12, len(FEATURE_KEYS)))
        if informative:                                         # el rasgo 0 y 1 llevan la señal
            feats[:, 0] += 0.5 * lat[i]
            feats[:, 1] += 0.3 * lat[i]
        ref = (120 + 12 * lat[i], 78 + 7 * lat[i]) if informative else (120 + 12 * rng.normal(), 78 + 7 * rng.normal())
        out.append({"keys": FEATURE_KEYS, "feats": feats, "ref": np.array(ref), "windows": None, "t": None, "subject": ""})
    return out


def test_cross_validation_detects_signal_and_no_signal():
    good = cross_validate_sessions(_fake_sessions(14, True, 0), alpha=1.0)
    bad = cross_validate_sessions(_fake_sessions(14, False, 1), alpha=1.0)
    assert good["sbp"]["beats_baseline"] and good["sbp"]["model"]["mae"] < 0.6 * good["sbp"]["baseline"]["mae"]
    assert not (bad["sbp"]["beats_baseline"] and bad["dbp"]["beats_baseline"]), "con etiquetas al azar no debe 'ganar'"
    print(f"  con señal: MAE modelo {good['sbp']['model']['mae']:.1f} vs trivial {good['sbp']['baseline']['mae']:.1f} mmHg"
          f" | sin señal: modelo {bad['sbp']['model']['mae']:.1f} vs trivial {bad['sbp']['baseline']['mae']:.1f} "
          f"(supera: SBP {bad['sbp']['beats_baseline']}, DBP {bad['dbp']['beats_baseline']})")
    try:
        cross_validate_sessions(_fake_sessions(3, True, 2))
        raise AssertionError("debió exigir más sesiones")
    except ValueError:
        pass


def test_torch_adapter_input_contract_without_torch():
    class Dummy:
        def eval(self): return self
        def to(self, d): return self
    w = np.linspace(0, 1, 750).astype(np.float32)
    for ch in (1, 2, 3):
        x = TorchBPModel(Dummy(), channels=ch)._stack(w)
        assert x.shape == (1, ch, 750) and x.dtype == np.float32 and x.min() >= 0 and x.max() <= 1 + 1e-6
    print("  adaptador PyTorch: entrada (1, C, 750) para C=1,2,3 (el forward NO se probó: sin PyTorch aquí)")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            print(name)
            fn()
    print("OK")
