"""rPPG modular: FC, FR y PA (investigación) con cámara RGB; SpO2 experimental."""
from .bp import (BloodPressureEstimator, BPCalibration, RidgeFeatureModel, TorchBPModel,
                 cross_validate_sessions, evaluate_bp, fit_all, load_session, save_session)
from .chest import ChestMotionTracker
from .config import PipelineConfig
from .estimators import (HeartRateEstimator, VitalEstimator, VitalResult, WindowContext)
from .pipeline import RPPGPipeline, Snapshot
from .quality import QualityReport
from .respiration import RespiratoryRateEstimator
from .spo2 import SpO2Calibration, SpO2Estimator, fit_calibration

__all__ = [
    "PipelineConfig", "RPPGPipeline", "Snapshot", "QualityReport", "ChestMotionTracker",
    "VitalEstimator", "VitalResult", "WindowContext",
    "HeartRateEstimator", "RespiratoryRateEstimator",
    "BloodPressureEstimator", "BPCalibration", "RidgeFeatureModel", "TorchBPModel",
    "evaluate_bp", "save_session", "load_session", "cross_validate_sessions", "fit_all",
    "SpO2Estimator", "SpO2Calibration", "fit_calibration",
]
