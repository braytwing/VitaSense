"""Presión arterial (PA) a partir de la onda de pulso rPPG — módulo de INVESTIGACIÓN.

Esto NO es un medidor de PA. Es el andamiaje para estudiar, con tus propios datos y un
tensiómetro de referencia, si un modelo aporta algo. Piezas:

  make_bp_window   ventana de 6 s a 125 Hz (750 muestras): el contrato de entrada de los modelos
                   del repo remotebiosensing/rppg (cnibp/), entrenados con PPG de contacto
  pulse_features   rasgos morfológicos de la plantilla de latido (+ FC, variabilidad)
  BPModel          interfaz: predict(feats, window) -> (sbp, dbp) sin calibrar
  RidgeFeatureModel   regresión ridge (solo numpy) entrenable con tus sesiones etiquetadas
  TorchBPModel     adaptador para un modelo PyTorch (p. ej. bvp2abp reentrenado). SIN PROBAR aquí
  BPCalibration    desfase por sujeto a partir de lecturas de tensiómetro
  BloodPressureEstimator   plug-in del pipeline: no da un número sin modelo Y calibración
  bp_metrics / evaluate_bp  error medio ± DE, MAE, r y comparación con un predictor trivial

Advertencias de fondo:
  - Los pesos incluidos en el repo (UCI, PPG de contacto de pacientes UCI) no son cargables con su
    código actual, y aunque lo fueran habría una brecha de dominio enorme: la onda rPPG de una
    webcam a 30 fps (≈25 muestras por latido, pasa-banda) no conserva la morfología fina.
  - Sin calibración por sujeto y validación externa, un modelo PPG->PA suele devolver
    esencialmente la media de la población. Mide siempre contra el predictor trivial.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Callable, Optional

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import find_peaks

from .estimators import VitalEstimator, VitalResult, WindowContext

BP_FS = 125.0           # Hz (datasets UCI/MIMIC)
BP_WIN = 750            # muestras = 6 s a 125 Hz
FEATURE_KEYS = ("hr_bpm", "ibi_sdnn_s", "t_peak_frac", "width50_frac", "max_slope", "area_before_peak")
EPS = 1e-12


# ----------------------------------------------------------------------------
# Ventana y rasgos
# ----------------------------------------------------------------------------
def make_bp_window(pulse: np.ndarray, fs: float, n: int = BP_WIN, fs_out: float = BP_FS):
    """Últimos n/fs_out s de la señal de pulso, a fs_out (spline cúbico), normalizada min-max."""
    m = int(round(n / fs_out * fs))
    if len(pulse) < m or m < 8:
        return None
    seg = np.asarray(pulse[-m:], float)
    t_in = np.arange(m) / fs
    x = CubicSpline(t_in, seg)(np.linspace(0.0, t_in[-1], n))
    rng = x.max() - x.min()
    if rng < EPS:
        return None
    return ((x - x.min()) / rng).astype(np.float32)


def pulse_features(pulse: np.ndarray, fs: float) -> Optional[dict]:
    """Rasgos de la plantilla de latido (mediana de latidos valle a valle, 100 puntos)."""
    pulse = np.asarray(pulse, float)
    peaks, _ = find_peaks(pulse, distance=max(int(0.4 * fs), 1), prominence=0.3 * pulse.std())
    if len(peaks) < 5:
        return None
    troughs = [a + int(np.argmin(pulse[a:b])) for a, b in zip(peaks[:-1], peaks[1:])]
    beats = []
    for k in range(1, len(peaks) - 1):                       # latido k: valle k-1 -> valle k
        seg = pulse[troughs[k - 1]:troughs[k] + 1]
        if len(seg) < 6 or np.ptp(seg) < EPS:
            continue
        seg = np.interp(np.linspace(0, len(seg) - 1, 100), np.arange(len(seg)), seg)
        beats.append((seg - seg.min()) / (seg.max() - seg.min()))
    if len(beats) < 3:
        return None
    tpl = np.median(np.array(beats), axis=0)
    ipk = int(np.argmax(tpl))
    ibi = np.diff(peaks) / fs
    return {"hr_bpm": float(60.0 / ibi.mean()),
            "ibi_sdnn_s": float(ibi.std()),
            "t_peak_frac": ipk / 99.0,
            "width50_frac": float((tpl >= 0.5).mean()),
            "max_slope": float(np.max(np.diff(tpl)) * 99.0),
            "area_before_peak": float(tpl[:ipk + 1].sum() / (tpl.sum() + EPS))}


# ----------------------------------------------------------------------------
# Modelos
# ----------------------------------------------------------------------------
class BPModel:
    """predict(feats, window) -> (sbp, dbp) en mmHg, SIN calibrar."""

    def predict(self, feats: dict, window: Optional[np.ndarray]) -> tuple[float, float]:
        raise NotImplementedError


class RidgeFeatureModel(BPModel):
    """Ridge sobre FEATURE_KEYS (estandarizados). Solo numpy; entrénalo con tus sesiones."""

    def __init__(self, keys=FEATURE_KEYS):
        self.keys = tuple(keys)
        self.mu = self.sd = self.W = self.b = None

    def matrix(self, feats_list) -> np.ndarray:
        if isinstance(feats_list, np.ndarray):          # ya es una matriz (n, len(keys))
            return feats_list.astype(float)
        return np.array([[f[k] for k in self.keys] for f in feats_list], float)

    def fit(self, feats_list, y, alpha: float = 1.0) -> "RidgeFeatureModel":
        X, y = self.matrix(feats_list), np.asarray(y, float)         # y: (n, 2) = [sbp, dbp]
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-9
        Z = (X - self.mu) / self.sd
        self.b = y.mean(0)
        self.W = np.linalg.solve(Z.T @ Z + alpha * np.eye(Z.shape[1]), Z.T @ (y - self.b))
        return self

    def predict_matrix(self, X: np.ndarray) -> np.ndarray:
        return self.b + ((X - self.mu) / self.sd) @ self.W

    def predict(self, feats, window=None):
        out = self.predict_matrix(self.matrix([feats]))[0]
        return float(out[0]), float(out[1])

    def save(self, path: str) -> None:
        np.savez(path, keys=np.array(self.keys), mu=self.mu, sd=self.sd, W=self.W, b=self.b)

    @classmethod
    def load(cls, path: str) -> "RidgeFeatureModel":
        d = np.load(path, allow_pickle=False)
        m = cls(tuple(str(k) for k in d["keys"]))
        m.mu, m.sd, m.W, m.b = d["mu"], d["sd"], d["W"], d["b"]
        return m


class TorchBPModel(BPModel):
    """Adaptador para un modelo PyTorch propio. NO PROBADO en este entorno (sin PyTorch).

    Contrato de entrada tomado del repo cnibp: tensor (1, C, 750); canales P (pulso), V (1ª derivada)
    y A (2ª derivada) según `channels` (1=P, 2=P+V, 3=P+V+A), cada uno normalizado min-max.
    Salida: el modelo bvp2abp devuelve (abp, dbp[15], sbp[15]) sin documentar cuál de los 15
    valores usar; por defecto se promedian. Pasa `parse_output` si tu modelo difiere.
    Úsalo con un modelo REENTRENADO por ti: los .pt del repo no son cargables (ver README).
    """

    def __init__(self, module, channels: int = 1, parse_output: Optional[Callable] = None,
                 device: str = "cpu"):
        self.module, self.channels, self.device = module.eval().to(device), channels, device
        self.parse_output = parse_output or self._default_parse

    @staticmethod
    def _default_parse(out):
        _, dbp, sbp = out
        return float(sbp.detach().cpu().numpy().mean()), float(dbp.detach().cpu().numpy().mean())

    def _stack(self, window: np.ndarray) -> np.ndarray:
        chans = [window]
        if self.channels >= 2:
            chans.append(np.gradient(window))
        if self.channels >= 3:
            chans.append(np.gradient(chans[1]))
        mm = lambda x: (x - x.min()) / (np.ptp(x) + EPS)               # noqa: E731
        return np.stack([mm(c) for c in chans])[None].astype(np.float32)

    def predict(self, feats, window):
        import torch
        with torch.no_grad():
            x = torch.from_numpy(self._stack(window)).to(self.device)
            return self.parse_output(self.module(x))


# ----------------------------------------------------------------------------
# Calibración por sujeto
# ----------------------------------------------------------------------------
@dataclass
class BPCalibration:
    """Desfase por sujeto: PA = predicción + offset. Una lectura basta para el offset; la
    pendiente no se puede estimar sin un rango amplio de PA, así que no se ajusta."""
    sbp_offset: float = 0.0
    dbp_offset: float = 0.0
    n: int = 0

    def apply(self, sbp: float, dbp: float) -> tuple[float, float]:
        return sbp + self.sbp_offset, dbp + self.dbp_offset

    @classmethod
    def fit(cls, preds, cuff) -> "BPCalibration":
        d = np.asarray(cuff, float) - np.asarray(preds, float)
        return cls(float(d[:, 0].mean()), float(d[:, 1].mean()), len(d))

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(asdict(self), fh, indent=2)

    @classmethod
    def load(cls, path: str) -> "BPCalibration":
        with open(path, encoding="utf-8") as fh:
            return cls(**json.load(fh))


# ----------------------------------------------------------------------------
# Estimador (plug-in del pipeline)
# ----------------------------------------------------------------------------
class BloodPressureEstimator(VitalEstimator):
    """No devuelve un número sin modelo Y calibración. `collect=True` guarda ventanas y rasgos
    para entrenar/evaluar después (ver save_session / train_bp.py)."""
    name = "bp"
    unit = "mmHg"

    def __init__(self, model: Optional[BPModel] = None, calibration: Optional[BPCalibration] = None,
                 collect: bool = False, min_hr_quality: float = 0.5):
        self.model, self.calibration = model, calibration
        self.collect, self.min_hr_quality = collect, min_hr_quality
        self.records: list = []          # sesión recolectada
        self.raw_preds: list = []        # predicciones sin calibrar (para calibrar contra un tensiómetro)

    def update(self, ctx: WindowContext) -> VitalResult:
        hr = ctx.results.get("hr")
        if hr is None or hr.value is None or hr.quality < self.min_hr_quality:
            return VitalResult("bp", None, self.unit, 0.0, {"note": "requiere una FC fiable"})
        feats = pulse_features(ctx.pulse, ctx.fs)
        win = make_bp_window(ctx.pulse, ctx.fs)
        if feats is None or win is None:
            return VitalResult("bp", None, self.unit, 0.0, {"note": "pocos latidos limpios"})
        if self.collect:
            self.records.append({"t": float(ctx.t[-1]), "feats": feats, "window": win,
                                 "hr_quality": float(hr.quality)})
        if self.model is None:
            return VitalResult("bp", None, self.unit, 0.0,
                               {"features": feats, "note": "sin modelo" + (" (recolectando)" if self.collect else "")})
        sbp, dbp = self.model.predict(feats, win)
        self.raw_preds.append((sbp, dbp))
        extra = {"raw_sbp": sbp, "raw_dbp": dbp, "features": feats, "calibrated": False}
        if self.calibration is None:
            return VitalResult("bp", None, self.unit, 0.0,
                               {**extra, "note": "sin calibrar: requiere lecturas de tensiómetro"})
        sbp_c, dbp_c = self.calibration.apply(sbp, dbp)
        extra.update(sbp=float(np.clip(sbp_c, 60, 260)), dbp=float(np.clip(dbp_c, 30, 160)), calibrated=True)
        return VitalResult("bp", extra["sbp"], self.unit, 0.5 * hr.quality, extra)


# ----------------------------------------------------------------------------
# Sesiones y evaluación
# ----------------------------------------------------------------------------
def save_session(path: str, records: list, ref: Optional[tuple] = None, subject: str = "") -> None:
    """Guarda una sesión recolectada (+ lectura de tensiómetro `ref=(sbp, dbp)`) en .npz."""
    if not records:
        raise ValueError("Sesión vacía")
    X = np.array([[r["feats"][k] for k in FEATURE_KEYS] for r in records], float)
    np.savez(path, keys=np.array(FEATURE_KEYS), feats=X,
             windows=np.stack([r["window"] for r in records]).astype(np.float32),
             t=np.array([r["t"] for r in records]),
             ref=np.array(ref if ref is not None else (np.nan, np.nan), float), subject=subject)


def load_session(path: str) -> dict:
    d = np.load(path, allow_pickle=False)
    return {"keys": tuple(str(k) for k in d["keys"]), "feats": d["feats"], "windows": d["windows"],
            "t": d["t"], "ref": d["ref"], "subject": str(d["subject"])}


def bp_metrics(pred, ref) -> dict:
    """Error medio ± DE (convención de validación de tensiómetros), MAE y correlación."""
    e = np.asarray(pred, float) - np.asarray(ref, float)
    r = float(np.corrcoef(pred, ref)[0, 1]) if len(e) > 2 and np.std(pred) > 0 and np.std(ref) > 0 else float("nan")
    me, sd = float(e.mean()), float(e.std(ddof=1)) if len(e) > 1 else float("nan")
    return {"n": len(e), "me": me, "sd": sd, "mae": float(np.abs(e).mean()), "r": r,
            "within_5_8": bool(abs(me) <= 5 and sd <= 8)}   # criterio de referencia 5 ± 8 mmHg (ISO 81060-2)


def evaluate_bp(y_true, y_pred, y_baseline) -> dict:
    """Compara el modelo con un predictor trivial (p. ej. la media del entrenamiento)."""
    out = {}
    for j, nm in enumerate(("sbp", "dbp")):
        t, p, b = (np.asarray(a, float)[:, j] for a in (y_true, y_pred, y_baseline))
        m, base = bp_metrics(p, t), bp_metrics(b, t)
        out[nm] = {"model": m, "baseline": base, "beats_baseline": bool(m["mae"] < base["mae"])}
    return out


def _valid_sessions(sessions: list):
    refs = np.array([s["ref"] for s in sessions], float)
    ok = np.isfinite(refs).all(axis=1)
    return [s for s, o in zip(sessions, ok) if o], refs[ok]


def _stack_xy(sessions, refs, idx):
    X = np.concatenate([sessions[j]["feats"] for j in idx])
    y = np.concatenate([np.repeat(refs[j][None], len(sessions[j]["feats"]), 0) for j in idx])
    return X, y


def cross_validate_sessions(sessions: list, alpha: float = 1.0) -> dict:
    """Leave-one-session-out. La etiqueta (tensiómetro) es una por sesión, así que se evalúa a nivel
    de sesión (media de las ventanas): las ventanas de una misma sesión están muy correlacionadas.
    Compara siempre con el predictor trivial 'media del entrenamiento'."""
    sessions, refs = _valid_sessions(sessions)
    if len(sessions) < 4:
        raise ValueError("Se necesitan al menos 4 sesiones con lectura de tensiómetro (idealmente >= 10, "
                         "en momentos y condiciones distintas)")
    keys = sessions[0]["keys"]
    preds, base = [], []
    for i in range(len(sessions)):
        X, y = _stack_xy(sessions, refs, [j for j in range(len(sessions)) if j != i])
        m = RidgeFeatureModel(keys).fit(X, y, alpha)
        preds.append(m.predict_matrix(sessions[i]["feats"]).mean(0))
        base.append(y.mean(0))
    res = evaluate_bp(refs, np.array(preds), np.array(base))
    res["n_sessions"] = len(sessions)
    res["ref_sd"] = {"sbp": float(refs[:, 0].std(ddof=1)), "dbp": float(refs[:, 1].std(ddof=1))}
    return res


def fit_all(sessions: list, alpha: float = 1.0) -> RidgeFeatureModel:
    sessions, refs = _valid_sessions(sessions)
    X, y = _stack_xy(sessions, refs, range(len(sessions)))
    return RidgeFeatureModel(sessions[0]["keys"]).fit(X, y, alpha)
