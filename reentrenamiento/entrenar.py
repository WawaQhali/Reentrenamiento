"""Pasos 7 y 8 del pipeline: reentrenar con las etiquetas del medico y proponer un candidato.

1. Junta las imagenes con etiqueta final (del backend) y, si hay, un conjunto base publico en
   `modelos/datos-base/{clase}/*.jpg`. Las `no_evaluable` no se usan.
2. Ajuste fino de MobileNetV3Large (ImageNet) en CPU, con validacion estratificada (20 %).
3. Exporta a TFLite y comprueba que el .tflite da probabilidades validas (sin NaN).
4. Sube `modelos/{version}/modelo.tflite` y `reporte.json`, registra el candidato y marca como
   usadas las imagenes donadas. Publicarlo lo decide el equipo: ahi actua la compuerta.
"""

import hashlib
import io
import json
import logging
import random
import re
from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np
from PIL import Image

from reentrenamiento.almacen import Almacen
from reentrenamiento.backend import ClienteBackend
from reentrenamiento.config import Ajustes

log = logging.getLogger(__name__)

_VERSION = re.compile(r"^modelos/(\d+)\.(\d+)\.(\d+)/modelo\.tflite$")
DESDE_SIEMPRE = datetime(2000, 1, 1, tzinfo=UTC)


class DatosInsuficientes(Exception):
    pass


@dataclass
class Ejemplo:
    ruta: str
    clase: int
    id_revision: str | None  # None si viene del conjunto base


def siguiente_version(almacen: Almacen) -> str:
    versiones = [
        tuple(int(p) for p in m.groups())
        for b in almacen.listar("modelos/")
        if (m := _VERSION.match(b.ruta))
    ]
    if not versiones:
        return "1.0.0"
    mayor, menor, _ = max(versiones)
    return f"{mayor}.{menor + 1}.0"


def juntar_ejemplos(ajustes: Ajustes, almacen: Almacen, backend: ClienteBackend) -> list[Ejemplo]:
    indice = {c: i for i, c in enumerate(ajustes.clases)}
    ejemplos = [
        Ejemplo(e["blob_path"], indice[e["etiqueta_final"]], str(e["id"]))
        for e in backend.etiquetas(DESDE_SIEMPRE)
        if e["etiqueta_final"] in indice
    ]
    for clase, i in indice.items():
        for blob in almacen.listar(f"{ajustes.prefijo_datos_base}/{clase}/"):
            if blob.ruta.lower().endswith((".jpg", ".jpeg", ".png")):
                ejemplos.append(Ejemplo(blob.ruta, i, None))
    return ejemplos


def _cargar(almacen: Almacen, ruta: str, lado: int) -> np.ndarray:
    with Image.open(io.BytesIO(almacen.leer(ruta))) as imagen:
        rgb = imagen.convert("RGB")
        ancho, alto = rgb.size
        corte = min(ancho, alto)  # recorte central cuadrado, como la app (seccion 5.3)
        izq, arriba = (ancho - corte) // 2, (alto - corte) // 2
        rgb = rgb.crop((izq, arriba, izq + corte, arriba + corte)).resize((lado, lado))
        return np.asarray(rgb, dtype=np.float32)  # 0..255: la normalizacion va en el modelo


def dividir(ejemplos: list[Ejemplo], semilla: int) -> tuple[list[Ejemplo], list[Ejemplo]]:
    """80/20 estratificado: al menos un ejemplo de cada clase en validacion."""
    azar = random.Random(semilla)  # noqa: S311 (solo baraja los ejemplos)
    entrenamiento, validacion = [], []
    for clase in sorted({e.clase for e in ejemplos}):
        grupo = [e for e in ejemplos if e.clase == clase]
        azar.shuffle(grupo)
        corte = max(1, round(len(grupo) * 0.2))
        validacion += grupo[:corte]
        entrenamiento += grupo[corte:]
    return entrenamiento, validacion


def metricas(reales: np.ndarray, predichas: np.ndarray, positiva: int, n_clases: int) -> dict:
    confusion = np.zeros((n_clases, n_clases), dtype=int)
    for r, p in zip(reales, predichas, strict=True):
        confusion[r, p] += 1
    positivos = confusion[positiva].sum()
    negativos = confusion.sum() - positivos
    verdaderos_neg = negativos - (confusion[:, positiva].sum() - confusion[positiva, positiva])
    return {
        "sensibilidad": float(confusion[positiva, positiva] / positivos) if positivos else 0.0,
        "especificidad": float(verdaderos_neg / negativos) if negativos else 0.0,
        "exactitud": float(np.trace(confusion) / confusion.sum()),
        "matriz_confusion": confusion.tolist(),
    }


def _modelo(ajustes: Ajustes):  # noqa: ANN202
    import tensorflow as tf

    lado = ajustes.tamano_entrada
    base = tf.keras.applications.MobileNetV3Large(
        input_shape=(lado, lado, 3),
        include_top=False,
        weights=None if ajustes.pesos == "ninguno" else ajustes.pesos,
        include_preprocessing=True,  # recibe 0..255, como dice el contrato (0_255)
    )
    base.trainable = False
    entrada = tf.keras.Input((lado, lado, 3))
    x = tf.keras.layers.RandomFlip("horizontal_and_vertical")(entrada)
    x = tf.keras.layers.RandomRotation(0.1)(x)
    x = tf.keras.layers.RandomContrast(0.15)(x)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    salida = tf.keras.layers.Dense(len(ajustes.clases), activation="softmax")(x)
    modelo = tf.keras.Model(entrada, salida)
    modelo.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return modelo


def a_tflite(modelo, muestra: np.ndarray) -> bytes:  # noqa: ANN001
    """Convierte y comprueba con el interprete que la salida es una distribucion valida."""
    import tensorflow as tf

    tflite = tf.lite.TFLiteConverter.from_keras_model(modelo).convert()
    interprete = tf.lite.Interpreter(model_content=tflite)
    interprete.allocate_tensors()
    entrada = interprete.get_input_details()[0]
    salida = interprete.get_output_details()[0]
    interprete.set_tensor(entrada["index"], muestra[:1].astype(np.float32))
    interprete.invoke()
    probabilidades = interprete.get_tensor(salida["index"])[0]
    if not np.all(np.isfinite(probabilidades)) or abs(float(probabilidades.sum()) - 1) > 1e-3:
        raise RuntimeError("El .tflite exportado no da probabilidades validas")
    return tflite


def ejecutar(ajustes: Ajustes, almacen: Almacen, backend: ClienteBackend) -> dict | None:
    import tensorflow as tf

    tf.keras.utils.set_random_seed(ajustes.semilla)
    ejemplos = juntar_ejemplos(ajustes, almacen, backend)
    por_clase = {c: sum(1 for e in ejemplos if e.clase == i) for i, c in enumerate(ajustes.clases)}
    log.info("Ejemplos por clase: %s", por_clase)
    if min(por_clase.values()) < ajustes.min_por_clase:
        log.warning(
            "Datos insuficientes (minimo %d por clase): no se entrena", ajustes.min_por_clase
        )
        return None

    entrenamiento, validacion = dividir(ejemplos, ajustes.semilla)
    lado = ajustes.tamano_entrada
    x_ent = np.stack([_cargar(almacen, e.ruta, lado) for e in entrenamiento])
    y_ent = np.array([e.clase for e in entrenamiento])
    x_val = np.stack([_cargar(almacen, e.ruta, lado) for e in validacion])
    y_val = np.array([e.clase for e in validacion])

    conteo = np.bincount(y_ent, minlength=len(ajustes.clases))
    pesos = {i: len(y_ent) / (len(conteo) * n) for i, n in enumerate(conteo) if n}
    modelo = _modelo(ajustes)
    modelo.fit(
        x_ent,
        y_ent,
        validation_data=(x_val, y_val),
        epochs=ajustes.epocas,
        batch_size=16,
        class_weight=pesos,
        verbose=2,
    )

    positiva = ajustes.clases.index(ajustes.clase_positiva)
    predichas = modelo.predict(x_val, verbose=0).argmax(axis=1)
    evaluacion = metricas(y_val, predichas, positiva, len(ajustes.clases))
    log.info("Validacion: %s", evaluacion)

    tflite = a_tflite(modelo, x_val)
    version = siguiente_version(almacen)
    ruta = f"modelos/{version}/modelo.tflite"
    sha = hashlib.sha256(tflite).hexdigest()
    usadas = [e.id_revision for e in entrenamiento + validacion if e.id_revision]
    reporte = {
        "version": version,
        "creado_en": datetime.now(UTC).isoformat(),
        "ejemplos_por_clase": por_clase,
        "entrenamiento": len(entrenamiento),
        "validacion": len(validacion),
        "imagenes_donadas": len(usadas),
        "epocas": ajustes.epocas,
        "base": f"MobileNetV3Large ({ajustes.pesos})",
        **evaluacion,
    }
    almacen.subir(ruta, tflite, "application/octet-stream", {"sha256": sha})
    almacen.subir(
        f"modelos/{version}/reporte.json",
        json.dumps(reporte, ensure_ascii=False, indent=1).encode(),
        "application/json",
        {},
    )
    candidato = {
        "version": version,
        "sensibilidad": round(evaluacion["sensibilidad"], 4),
        "blob_path": ruta,
        "sha256": sha,
        "tamano": len(tflite),
        "entrada": {
            "ancho": lado,
            "alto": lado,
            "canales": 3,
            "normalizacion": ajustes.normalizacion,
        },
        "clases": list(ajustes.clases),
        "umbral_confianza": ajustes.umbral_confianza,
    }
    backend.registrar_candidato(candidato)
    if usadas:
        backend.marcar_usadas(version, usadas)
    log.info("Candidato %s registrado (sensibilidad %.3f)", version, evaluacion["sensibilidad"])
    return reporte
