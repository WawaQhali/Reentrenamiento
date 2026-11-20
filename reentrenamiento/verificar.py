"""Pasos 4 y 6 del pipeline: de la cuarentena a la revision del medico.

Por cada paquete en `cuarentena/{installation_id}/{paquete_id}.json`:
descifra con Key Vault -> JPEG 1024 x 1024 -> sin rostro -> sin duplicado
-> sube el recorte a `verificadas/{paquete_id}.jpg` -> lo registra en Revision.
El paquete sale de la cuarentena cuando se registra o se descarta. Si falla algo pasajero
(Key Vault, red, backend caido) se deja para la siguiente corrida.
"""

import base64
import binascii
import hashlib
import json
import logging
import uuid
from collections import Counter
from dataclasses import dataclass, field

from reentrenamiento import imagen
from reentrenamiento.almacen import Almacen
from reentrenamiento.backend import ClienteBackend, YaRegistrado
from reentrenamiento.cifrado import Desenvolvedor, PaqueteInvalido, descifrar

log = logging.getLogger(__name__)

PREFIJO_CUARENTENA = "cuarentena/"
PREFIJO_VERIFICADAS = "verificadas/"


@dataclass
class Resumen:
    registradas: int = 0
    descartadas: Counter = field(default_factory=Counter)
    reintentar: int = 0

    def __str__(self) -> str:
        descartes = ", ".join(f"{m}={n}" for m, n in sorted(self.descartadas.items())) or "0"
        return (
            f"registradas={self.registradas} descartadas[{descartes}] "
            f"para_reintentar={self.reintentar}"
        )


def _hash_instalacion(installation_id: str) -> str:
    """Los logs nunca llevan el installation_id en claro (seccion 9)."""
    return hashlib.sha256(installation_id.encode()).hexdigest()[:12]


def _partes(ruta: str) -> tuple[uuid.UUID, uuid.UUID]:
    _, instalacion, archivo = ruta.split("/", 2)
    if not archivo.endswith(".json"):
        raise ValueError(ruta)
    return uuid.UUID(instalacion), uuid.UUID(archivo.removesuffix(".json"))


def _huellas_existentes(almacen: Almacen) -> list[int]:
    huellas = []
    for blob in almacen.listar(PREFIJO_VERIFICADAS):
        if "dhash" in blob.metadatos:
            huellas.append(int(blob.metadatos["dhash"], 16))
    return huellas


def ejecutar(
    almacen: Almacen,
    desenvolvedor: Desenvolvedor,
    backend: ClienteBackend,
    distancia_duplicado: int = 5,
) -> Resumen:
    resumen = Resumen()
    huellas = _huellas_existentes(almacen)
    sha_vistos: set[str] = set()

    for blob in almacen.listar(PREFIJO_CUARENTENA):
        try:
            instalacion, paquete_id = _partes(blob.ruta)
        except ValueError:
            log.warning("Ruta inesperada en la cuarentena, se ignora")
            continue
        etiqueta_log = f"paquete={paquete_id} instalacion#{_hash_instalacion(str(instalacion))}"

        def descartar(motivo: str, ruta: str = blob.ruta, etiqueta: str = etiqueta_log) -> None:
            almacen.borrar(ruta)
            resumen.descartadas[motivo] += 1
            log.info("Descartado %s: %s", etiqueta, motivo)

        try:
            paquete = json.loads(almacen.leer(blob.ruta))
            contenido = descifrar(paquete, instalacion, desenvolvedor)
            jpeg = base64.b64decode(contenido["imagen_jpeg"], validate=True)
            rgb = imagen.abrir_jpeg(jpeg)
        except (PaqueteInvalido, imagen.ImagenRechazada) as error:
            descartar(error.motivo)
            continue
        except (ValueError, binascii.Error):
            descartar("formato")
            continue
        except Exception as error:  # Key Vault o Blob con problemas: se reintenta despues
            resumen.reintentar += 1
            log.warning("No se pudo procesar %s: %s", etiqueta_log, type(error).__name__)
            continue

        if imagen.hay_rostro(rgb):
            descartar("rostro")
            continue
        sha = hashlib.sha256(jpeg).hexdigest()
        huella = imagen.dhash(rgb)
        if sha in sha_vistos or any(
            imagen.distancia(huella, h) <= distancia_duplicado for h in huellas
        ):
            descartar("duplicado")
            continue

        ruta_verificada = f"{PREFIJO_VERIFICADAS}{paquete_id}.jpg"
        try:
            almacen.subir(
                ruta_verificada, jpeg, "image/jpeg", {"dhash": f"{huella:016x}", "sha256": sha}
            )
            backend.registrar_verificada(
                str(paquete_id),
                str(instalacion),
                ruta_verificada,
                contenido["etiqueta_preliminar"],
            )
        except YaRegistrado:
            log.info("Ya estaba registrado %s", etiqueta_log)
        except Exception as error:
            # 422 (etiqueta invalida) no se arregla reintentando: se descarta.
            codigo = getattr(getattr(error, "response", None), "status_code", None)
            almacen.borrar(ruta_verificada)
            if codigo == 422:
                descartar("etiqueta_invalida")
            else:
                resumen.reintentar += 1
                log.warning("No se pudo registrar %s: %s", etiqueta_log, type(error).__name__)
            continue

        almacen.borrar(blob.ruta)
        huellas.append(huella)
        sha_vistos.add(sha)
        resumen.registradas += 1
        log.info("Registrado %s", etiqueta_log)

    return resumen
