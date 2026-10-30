"""Fuentes de vídeo.

Para usar una cámara más avanzada (RealSense, industrial, NIR, etc.) basta con
heredar de `CameraSource` y devolver `Frame` con:
  - image: BGR uint8 (H, W, 3)  -> es lo que consumen los detectores de cara
  - t: timestamp en segundos (ideal: el de hardware, no el de llegada a Python)
  - meta: dict libre (exposición, ganancia, canal NIR, profundidad, etc.)
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np


@dataclass
class Frame:
    image: np.ndarray
    t: float
    index: int = 0
    meta: dict = field(default_factory=dict)


class CameraSource(ABC):
    nominal_fps: float = 30.0

    @abstractmethod
    def open(self) -> None: ...

    @abstractmethod
    def read(self) -> Optional[Frame]:
        """Devuelve el siguiente Frame o None si se acabó / falló."""

    @abstractmethod
    def close(self) -> None: ...

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *exc):
        self.close()


class WebcamSource(CameraSource):
    """Webcam estándar vía OpenCV.

    `props` permite pasar propiedades específicas de tu backend/cámara, p. ej.
    para fijar exposición manual (los valores cambian entre DSHOW, V4L2, MSMF):
        props={cv2.CAP_PROP_AUTO_EXPOSURE: 0.25, cv2.CAP_PROP_EXPOSURE: -6}
    Fijar exposición y balance de blancos mejora mucho la señal rPPG.
    """

    def __init__(self, index: int = 0, width: int = 640, height: int = 480,
                 fps: float = 30.0, fourcc: Optional[str] = "MJPG",
                 backend: Optional[int] = None, props: Optional[dict] = None):
        self.index, self.width, self.height = index, width, height
        self.fps, self.fourcc, self.backend = fps, fourcc, backend
        self.props = props or {}
        self._cap: Optional[cv2.VideoCapture] = None
        self._i = 0

    def open(self) -> None:
        cap = (cv2.VideoCapture(self.index) if self.backend is None
               else cv2.VideoCapture(self.index, self.backend))
        if not cap.isOpened():
            raise RuntimeError(f"No se pudo abrir la cámara {self.index}")
        if self.fourcc:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.fourcc))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        cap.set(cv2.CAP_PROP_AUTO_WB, 0)  # muchas webcams lo ignoran; no pasa nada
        for k, v in self.props.items():
            cap.set(k, v)
        real = cap.get(cv2.CAP_PROP_FPS)
        self.nominal_fps = real if real and real > 0 else self.fps
        self._cap, self._i = cap, 0

    def read(self) -> Optional[Frame]:
        ok, img = self._cap.read()
        t = time.perf_counter()
        if not ok:
            return None
        f = Frame(img, t, self._i)
        self._i += 1
        return f

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None


class VideoFileSource(CameraSource):
    """Vídeo grabado (validación offline). El tiempo sale del índice de frame."""

    def __init__(self, path: str, fps: Optional[float] = None):
        self.path, self._fps_override = path, fps
        self._cap: Optional[cv2.VideoCapture] = None
        self._i = 0

    def open(self) -> None:
        cap = cv2.VideoCapture(self.path)
        if not cap.isOpened():
            raise RuntimeError(f"No se pudo abrir {self.path}")
        real = cap.get(cv2.CAP_PROP_FPS)
        self.nominal_fps = self._fps_override or (real if real and real > 0 else 30.0)
        self._cap, self._i = cap, 0

    def read(self) -> Optional[Frame]:
        ok, img = self._cap.read()
        if not ok:
            return None
        f = Frame(img, self._i / self.nominal_fps, self._i)
        self._i += 1
        return f

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
