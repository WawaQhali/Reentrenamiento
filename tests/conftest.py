import io
import uuid
from pathlib import Path

import httpx
import numpy as np
import pytest
from PIL import Image

from reentrenamiento.almacen import AlmacenMemoria
from reentrenamiento.backend import ClienteBackend
from reentrenamiento.cifrado import DesenvolvedorLocal

DATOS = Path(__file__).parent / "datos"

ETIQUETA = {
    "clase": "compatible",
    "confianza": 0.91,
    "respuestas": {"picazon": True, "extension": "una_zona"},
    "version_modelo": "2026-11-14-mobilenetv3",
    "version_cuestionario": "v1",
}


def jpeg(rgb: np.ndarray, calidad: int = 85) -> bytes:
    salida = io.BytesIO()
    Image.fromarray(rgb).save(salida, "JPEG", quality=calidad)
    return salida.getvalue()


def piel_con_ronchas(semilla: int, lado: int = 1024) -> np.ndarray:
    """Piel sintetica con ronchas rojizas: no tiene rostro."""
    azar = np.random.default_rng(semilla)
    base = np.array([224, 172, 140], dtype=np.float32)
    img = np.tile(base, (lado, lado, 1)) + azar.normal(0, 6, (lado, lado, 3))
    yy, xx = np.mgrid[0:lado, 0:lado]
    for _ in range(int(azar.integers(4, 9))):
        cy, cx, r = (
            azar.integers(100, lado - 100),
            azar.integers(100, lado - 100),
            azar.integers(40, 140),
        )
        mascara = (yy - cy) ** 2 + (xx - cx) ** 2 < r**2
        img[mascara] = img[mascara] * 0.6 + np.array([235, 110, 110]) * 0.4
    return np.clip(img, 0, 255).astype(np.uint8)


def rostro() -> np.ndarray:
    """Foto de dominio publico (NASA, astronauta Eileen Collins), llevada a 1024 x 1024."""
    with Image.open(DATOS / "astronauta.png") as imagen:
        return np.asarray(imagen.convert("RGB").resize((1024, 1024)))


class BackendFalso:
    """Simula las rutas internas del backend con httpx.MockTransport."""

    def __init__(self) -> None:
        self.verificadas: list[dict] = []
        self.candidatos: list[dict] = []
        self.usadas: list[dict] = []
        self.etiquetas: list[dict] = []
        self.falla_registro: int | None = None

    def manejar(self, peticion: httpx.Request) -> httpx.Response:
        import json

        assert peticion.headers["X-Api-Key"] == "clave-de-prueba"
        ruta = peticion.url.path
        if ruta == "/v1/interno/revision/verificadas":
            if self.falla_registro:
                return httpx.Response(self.falla_registro)
            cuerpo = json.loads(peticion.content)
            if any(v["paquete_id"] == cuerpo["paquete_id"] for v in self.verificadas):
                return httpx.Response(409)
            self.verificadas.append(cuerpo)
            return httpx.Response(201, json={"id": str(uuid.uuid4())})
        if ruta == "/v1/interno/revision/etiquetas":
            return httpx.Response(200, json=self.etiquetas)
        if ruta == "/v1/interno/modelos/candidatos":
            self.candidatos.append(json.loads(peticion.content))
            return httpx.Response(201, json={"version": "x", "estado": "candidato"})
        if ruta == "/v1/interno/revision/usadas":
            cuerpo = json.loads(peticion.content)
            self.usadas.append(cuerpo)
            return httpx.Response(200, json={"marcadas": len(cuerpo["ids"])})
        return httpx.Response(404)


@pytest.fixture
def almacen() -> AlmacenMemoria:
    return AlmacenMemoria()


@pytest.fixture(scope="session")
def claves() -> DesenvolvedorLocal:
    return DesenvolvedorLocal()


@pytest.fixture
def falso() -> BackendFalso:
    return BackendFalso()


@pytest.fixture
def backend(falso: BackendFalso) -> ClienteBackend:
    http = httpx.Client(base_url="http://backend", transport=httpx.MockTransport(falso.manejar))
    return ClienteBackend("http://backend", "clave-de-prueba", http)
