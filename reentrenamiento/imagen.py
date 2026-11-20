"""Controles sobre la imagen descifrada (secciones 5.5 y 6.4 de la guia del backend):
JPEG valido de 1024 x 1024, sin rostro, y una huella para detectar duplicados."""

import io

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

LADO = 1024

_CASCADAS = [
    cv2.CascadeClassifier(cv2.data.haarcascades + nombre)
    for nombre in (
        "haarcascade_frontalface_default.xml",
        "haarcascade_frontalface_alt2.xml",
        "haarcascade_profileface.xml",
    )
]


class ImagenRechazada(Exception):
    def __init__(self, motivo: str) -> None:
        super().__init__(motivo)
        self.motivo = motivo


def abrir_jpeg(datos: bytes) -> np.ndarray:
    """RGB uint8 de 1024 x 1024, o ImagenRechazada."""
    try:
        with Image.open(io.BytesIO(datos)) as imagen:
            if imagen.format != "JPEG":
                raise ImagenRechazada("no_es_jpeg")
            if imagen.size != (LADO, LADO):
                raise ImagenRechazada("tamano")
            return np.asarray(imagen.convert("RGB"))
    except (UnidentifiedImageError, OSError) as error:
        raise ImagenRechazada("jpeg_invalido") from error


def hay_rostro(rgb: np.ndarray) -> bool:
    """Detector de OpenCV (frontal y de perfil, en las dos orientaciones). Es el segundo
    control: el primero es Google ML Kit en el celular y el tercero, el medico."""
    gris = cv2.equalizeHist(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))
    gris = cv2.resize(gris, (512, 512))
    for variante in (gris, cv2.flip(gris, 1)):
        for cascada in _CASCADAS:
            caras = cascada.detectMultiScale(
                variante, scaleFactor=1.1, minNeighbors=6, minSize=(60, 60)
            )
            if len(caras) > 0:
                return True
    return False


def dhash(rgb: np.ndarray) -> int:
    """Huella perceptual de 64 bits: dos fotos casi iguales difieren en pocos bits."""
    gris = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    chica = cv2.resize(gris, (9, 8), interpolation=cv2.INTER_AREA)
    bits = (chica[:, 1:] > chica[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def distancia(a: int, b: int) -> int:
    return (a ^ b).bit_count()
