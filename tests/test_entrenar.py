import hashlib
import uuid

import numpy as np
import pytest

from reentrenamiento import entrenar
from reentrenamiento.config import Ajustes
from tests.conftest import ETIQUETA, jpeg, piel_con_ronchas


def test_siguiente_version(almacen) -> None:  # noqa: ANN001
    assert entrenar.siguiente_version(almacen) == "1.0.0"
    for v in ("1.0.0", "1.2.0", "1.10.0"):
        almacen.subir(f"modelos/{v}/modelo.tflite", b"x", "application/octet-stream", {})
    almacen.subir("modelos/datos-base/compatible/a.jpg", b"x", "image/jpeg", {})
    assert entrenar.siguiente_version(almacen) == "1.11.0"


def test_division_estratificada_deja_cada_clase_en_validacion() -> None:
    ejemplos = [entrenar.Ejemplo(f"r{i}", i % 2, None) for i in range(23)]
    ent, val = entrenar.dividir(ejemplos, semilla=1)
    assert len(ent) + len(val) == 23
    assert {e.clase for e in val} == {0, 1}
    assert not {e.ruta for e in ent} & {e.ruta for e in val}


def test_metricas_sensibilidad_y_especificidad() -> None:
    reales = np.array([0, 0, 0, 0, 1, 1])
    predichas = np.array([0, 0, 0, 1, 1, 0])
    m = entrenar.metricas(reales, predichas, positiva=0, n_clases=2)
    assert m["sensibilidad"] == 0.75
    assert m["especificidad"] == 0.5
    assert m["matriz_confusion"] == [[3, 1], [1, 1]]


@pytest.mark.entrenamiento
def test_entrena_exporta_y_registra_un_candidato(almacen, backend, falso) -> None:  # noqa: ANN001
    pytest.importorskip("tensorflow")
    ajustes = Ajustes(
        backend_url="http://backend", pesos="ninguno", epocas=1, min_por_clase=4, tamano_entrada=96
    )
    # 6 donadas etiquetadas por el medico (3 y 3) + 4 por clase del conjunto base
    for i in range(6):
        ruta = f"verificadas/{uuid.uuid4()}.jpg"
        almacen.subir(ruta, jpeg(piel_con_ronchas(i)), "image/jpeg", {})
        falso.etiquetas.append(
            {
                "id": str(uuid.uuid4()),
                "paquete_id": str(uuid.uuid4()),
                "blob_path": ruta,
                "etiqueta_final": "compatible" if i % 2 else "no_compatible",
                "etiqueta_preliminar": ETIQUETA,
                "revisada_en": "2026-11-20T10:00:00Z",
            }
        )
    falso.etiquetas.append(
        {**falso.etiquetas[0], "id": str(uuid.uuid4()), "etiqueta_final": "no_evaluable"}
    )
    for clase in ("compatible", "no_compatible"):
        for i in range(4):
            almacen.subir(
                f"modelos/datos-base/{clase}/{i}.jpg",
                jpeg(piel_con_ronchas(50 + i, 300)),
                "image/jpeg",
                {},
            )

    reporte = entrenar.ejecutar(ajustes, almacen, backend)

    assert reporte["version"] == "1.0.0"
    assert reporte["ejemplos_por_clase"] == {"compatible": 7, "no_compatible": 7}
    [candidato] = falso.candidatos
    tflite = almacen.leer("modelos/1.0.0/modelo.tflite")
    assert candidato["sha256"] == hashlib.sha256(tflite).hexdigest()
    assert candidato["tamano"] == len(tflite)
    assert candidato["clases"] == ["compatible", "no_compatible"]
    assert candidato["entrada"] == {"ancho": 96, "alto": 96, "canales": 3, "normalizacion": "0_255"}
    assert 0 <= candidato["sensibilidad"] <= 1
    [usadas] = falso.usadas
    assert usadas["version_modelo"] == "1.0.0" and len(usadas["ids"]) == 6  # sin la no_evaluable
    assert almacen.listar("modelos/1.0.0/reporte.json")


@pytest.mark.entrenamiento
def test_sin_datos_suficientes_no_entrena(almacen, backend, falso) -> None:  # noqa: ANN001
    pytest.importorskip("tensorflow")
    ajustes = Ajustes(backend_url="http://backend", pesos="ninguno")
    assert entrenar.ejecutar(ajustes, almacen, backend) is None
    assert falso.candidatos == []
