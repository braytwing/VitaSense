"""Detección de cara, definición de ROIs de piel y promedio RGB por frame."""
from __future__ import annotations

import os
import time
import urllib.request
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

MODEL_FILE = "face_landmarker.task"
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
             "face_landmarker/float16/1/face_landmarker.task")


def ensure_face_landmarker_model(path: str = MODEL_FILE) -> str:
    """Devuelve la ruta del modelo FaceLandmarker, descargándolo si no existe."""
    if os.path.exists(path):
        return path
    try:
        urllib.request.urlretrieve(MODEL_URL, path)
    except Exception as e:
        raise RuntimeError(
            f"No se pudo descargar el modelo. Bájalo manualmente de {MODEL_URL} "
            f"y guárdalo como '{path}'. Detalle: {e}") from e
    return path


@dataclass
class RoiResult:
    rgb: np.ndarray                      # (3,) media ponderada en orden R, G, B
    n_pixels: int
    sat_frac: float                      # fracción de píxeles saturados (>=250)
    nose_xy: tuple[float, float]         # para estimar movimiento
    face_width: float                    # px, para normalizar movimiento
    polygons: dict = field(default_factory=dict)   # nombre -> Nx2 int32 (overlay)
    per_roi: dict = field(default_factory=dict)    # nombre -> (3,) RGB medio
    face_box: Optional[tuple] = None     # (x, y, w, h) en px; ancla del ROI de tórax
    alar: Optional[float] = None         # ancho de las alas nasales / ancho de cara (experimental)


class ROIExtractor(ABC):
    @abstractmethod
    def process(self, frame_bgr: np.ndarray) -> Optional[RoiResult]: ...

    def close(self) -> None:
        pass


# ----------------------------------------------------------------------------
def _masked_mean(img: np.ndarray, poly: np.ndarray, skin_filter: bool, erode_px: int):
    """Media RGB dentro de un polígono. Devuelve (rgb, n, n_sat) o None."""
    h, w = img.shape[:2]
    poly = poly.astype(np.int32)
    x, y, bw, bh = cv2.boundingRect(poly)
    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + bw, w), min(y + bh, h)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None
    crop = img[y0:y1, x0:x1]
    mask = np.zeros(crop.shape[:2], np.uint8)
    cv2.fillPoly(mask, [poly - np.array([x0, y0], np.int32)], 255)
    if erode_px > 1:
        mask = cv2.erode(mask, np.ones((erode_px, erode_px), np.uint8))
    if skin_filter:  # descarta cejas, pelo, brillos, gafas...
        ycc = cv2.cvtColor(crop, cv2.COLOR_BGR2YCrCb)
        skin = cv2.inRange(ycc, np.array([0, 133, 77], np.uint8),
                           np.array([255, 173, 127], np.uint8))
        m2 = cv2.bitwise_and(mask, skin)
        if cv2.countNonZero(m2) >= 0.3 * max(cv2.countNonZero(mask), 1):
            mask = m2
    n = cv2.countNonZero(mask)
    if n < 30:
        return None
    b, g, r, _ = cv2.mean(crop, mask=mask)
    sat_mask = (crop.max(axis=2) >= 250).astype(np.uint8) * 255
    n_sat = cv2.countNonZero(cv2.bitwise_and(mask, sat_mask))
    return np.array([r, g, b]), n, n_sat


def _combine(img, polys: dict, skin_filter: bool, nose_xy, face_w: float,
             face_box=None, alar=None) -> Optional[RoiResult]:
    erode_px = max(3, int(0.015 * face_w)) | 1
    acc, total, sat, per = np.zeros(3), 0, 0, {}
    for name, poly in polys.items():
        out = _masked_mean(img, poly, skin_filter, erode_px)
        if out is None:
            continue
        rgb, n, n_sat = out
        acc += rgb * n
        total += n
        sat += n_sat
        per[name] = rgb
    if total == 0:
        return None
    return RoiResult(rgb=acc / total, n_pixels=total, sat_frac=sat / total,
                     nose_xy=(float(nose_xy[0]), float(nose_xy[1])),
                     face_width=float(face_w), polygons=polys, per_roi=per,
                     face_box=face_box, alar=alar)


# ----------------------------------------------------------------------------
class FaceMeshROI(ROIExtractor):
    """MediaPipe FaceMesh: frente (polígono) + mejillas (círculos)."""

    FOREHEAD = [103, 67, 109, 10, 338, 297, 332, 300, 293, 334, 296, 336,
                9, 107, 66, 105, 63, 70]
    CHEEKS = {"cheek_r": (50, 205), "cheek_l": (280, 425)}

    def __init__(self, rois=("forehead", "cheek_l", "cheek_r"),
                 ema_alpha: float = 0.5, skin_filter: bool = True,
                 model_path: str = MODEL_FILE):
        import mediapipe as mp
        self._mp = mp
        self.rois, self.ema_alpha, self.skin_filter = tuple(rois), ema_alpha, skin_filter
        self._prev: Optional[np.ndarray] = None
        self._last_ts = 0
        if hasattr(mp, "solutions"):            # mediapipe antiguo (API legacy)
            self._legacy = True
            self._mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False, max_num_faces=1, refine_landmarks=False,
                min_detection_confidence=0.5, min_tracking_confidence=0.5)
        else:                                   # mediapipe >= 0.10.2x: solo API Tasks
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision
            self._legacy = False
            opts = vision.FaceLandmarkerOptions(
                base_options=mp_python.BaseOptions(
                    model_asset_path=ensure_face_landmarker_model(model_path)),
                running_mode=vision.RunningMode.VIDEO, num_faces=1,
                min_face_detection_confidence=0.5, min_tracking_confidence=0.5)
            self._mesh = vision.FaceLandmarker.create_from_options(opts)

    def _landmarks_norm(self, frame_bgr) -> Optional[np.ndarray]:
        """(N, 2) landmarks normalizados [0,1] o None si no hay cara."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        if self._legacy:
            res = self._mesh.process(rgb)
            lms = res.multi_face_landmarks[0].landmark if res.multi_face_landmarks else None
        else:
            img = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
            self._last_ts = max(int(time.perf_counter() * 1000), self._last_ts + 1)  # ms, creciente
            res = self._mesh.detect_for_video(img, self._last_ts)
            lms = res.face_landmarks[0] if res.face_landmarks else None
        if lms is None:
            return None
        return np.array([(p.x, p.y) for p in lms])

    def process(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        norm = self._landmarks_norm(frame_bgr)
        if norm is None:
            self._prev = None
            return None
        pts = norm * np.array([w, h])
        if self._prev is not None:      # EMA: reduce el jitter de landmarks (ruido en la señal)
            pts = self.ema_alpha * pts + (1 - self.ema_alpha) * self._prev
        self._prev = pts
        fw = float(np.linalg.norm(pts[234] - pts[454]))
        polys = {}
        if "forehead" in self.rois:
            polys["forehead"] = pts[self.FOREHEAD].astype(np.int32)
        for name, ids in self.CHEEKS.items():
            if name in self.rois:
                c = pts[list(ids)].mean(axis=0)
                r = max(int(0.09 * fw), 4)
                polys[name] = cv2.ellipse2Poly((int(c[0]), int(c[1])), (r, r), 0, 0, 360, 20)
        x0, y0 = pts.min(axis=0)
        x1, y1 = pts.max(axis=0)
        alar = float(np.linalg.norm(pts[129] - pts[358]) / max(fw, 1e-6))
        return _combine(frame_bgr, polys, self.skin_filter, pts[1], fw,
                        face_box=(float(x0), float(y0), float(x1 - x0), float(y1 - y0)), alar=alar)

    def close(self):
        self._mesh.close()


class HaarROI(ROIExtractor):
    """Fallback sin mediapipe: Haar + ROIs rectangulares proporcionales (menos preciso)."""

    def __init__(self, ema_alpha: float = 0.3, skin_filter: bool = True):
        self._cc = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.ema_alpha, self.skin_filter = ema_alpha, skin_filter
        self._box: Optional[np.ndarray] = None

    @staticmethod
    def _rect(x0, y0, x1, y1):
        return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.int32)

    def process(self, frame_bgr):
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = self._cc.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))
        if len(faces) == 0:
            self._box = None
            return None
        box = np.array(max(faces, key=lambda f: f[2] * f[3]), float)
        a = self.ema_alpha
        self._box = box if self._box is None else a * box + (1 - a) * self._box
        x, y, w, h = self._box
        polys = {
            "forehead": self._rect(x + .28 * w, y + .05 * h, x + .72 * w, y + .22 * h),
            "cheek_r": self._rect(x + .12 * w, y + .50 * h, x + .32 * w, y + .68 * h),
            "cheek_l": self._rect(x + .68 * w, y + .50 * h, x + .88 * w, y + .68 * h),
        }
        return _combine(frame_bgr, polys, self.skin_filter, (x + w / 2, y + .6 * h), w,
                        face_box=(float(x), float(y), float(w), float(h)))


def make_roi_extractor(kind: str = "auto") -> ROIExtractor:
    """kind: 'auto' | 'mesh' | 'haar'."""
    if kind in ("auto", "mesh"):
        try:
            return FaceMeshROI()
        except Exception as e:  # ImportError, RuntimeError...
            if kind == "mesh":
                raise
            warnings.warn(f"FaceMesh no disponible ({e}); usando Haar (menos preciso).")
    return HaarROI()
