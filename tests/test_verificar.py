import base64
import json
import uuid

from reentrenamiento import verificar
from tests.conftest import ETIQUETA, jpeg, piel_con_ronchas, rostro


def _donar(almacen, claves, rgb, instalacion=None, carpeta=None) -> tuple[uuid.UUID, uuid.UUID]:  # noqa: ANN001
    """Deja en la cuarentena un paquete como lo guarda el backend."""
    instalacion = instalacion or uuid.uuid4()
    paquete_id = uuid.uuid4()
    contenido = {
        "imagen_jpeg": base64.b64encode(jpeg(rgb)).decode(),
        "etiqueta_preliminar": ETIQUETA,
    }
    paquete = claves.cifrar_para_pruebas(str(instalacion), contenido)
    ruta = f"cuarentena/{carpeta or instalacion}/{paquete_id}.json"
    almacen.subir(ruta, json.dumps(paquete).encode(), "application/json", {})
    return instalacion, paquete_id


def _cuarentena(almacen) -> list[str]:  # noqa: ANN001
    return [b.ruta for b in almacen.listar("cuarentena/")]


def test_paquete_valido_se_registra_y_sale_de_la_cuarentena(
    almacen, claves, backend, falso
) -> None:  # noqa: ANN001
    instalacion, paquete_id = _donar(almacen, claves, piel_con_ronchas(1))

    resumen = verificar.ejecutar(almacen, claves, backend)

    assert resumen.registradas == 1
    assert _cuarentena(almacen) == []
    assert falso.verificadas == [
        {
            "paquete_id": str(paquete_id),
            "installation_id": str(instalacion),
            "blob_path": f"verificadas/{paquete_id}.jpg",
            "etiqueta_preliminar": ETIQUETA,
        }
    ]
    [blob] = almacen.listar("verificadas/")
    assert set(blob.metadatos) == {"dhash", "sha256"}


def test_con_rostro_se_descarta_sin_registrar(almacen, claves, backend, falso) -> None:  # noqa: ANN001
    _donar(almacen, claves, rostro())
    resumen = verificar.ejecutar(almacen, claves, backend)
    assert resumen.descartadas == {"rostro": 1}
    assert falso.verificadas == []
    assert _cuarentena(almacen) == [] and almacen.listar("verificadas/") == []


def test_duplicados_se_descartan_en_la_misma_corrida_y_en_las_siguientes(
    almacen, claves, backend
) -> None:  # noqa: ANN001
    foto = piel_con_ronchas(2)
    _donar(almacen, claves, foto)
    _donar(almacen, claves, foto)
    assert verificar.ejecutar(almacen, claves, backend).registradas == 1
    _donar(almacen, claves, foto)
    resumen = verificar.ejecutar(almacen, claves, backend)
    assert resumen.registradas == 0 and resumen.descartadas == {"duplicado": 1}


def test_paquete_movido_a_otra_instalacion_no_se_descifra(almacen, claves, backend) -> None:  # noqa: ANN001
    _donar(almacen, claves, piel_con_ronchas(1), carpeta=uuid.uuid4())
    resumen = verificar.ejecutar(almacen, claves, backend)
    assert resumen.descartadas == {"aad_o_datos_alterados": 1}


def test_si_el_backend_falla_el_paquete_queda_para_la_siguiente_corrida(
    almacen, claves, backend, falso
) -> None:  # noqa: ANN001
    _donar(almacen, claves, piel_con_ronchas(1))
    falso.falla_registro = 503
    resumen = verificar.ejecutar(almacen, claves, backend)
    assert resumen.reintentar == 1
    assert len(_cuarentena(almacen)) == 1
    assert almacen.listar("verificadas/") == []  # no queda un recorte huerfano

    falso.falla_registro = None
    assert verificar.ejecutar(almacen, claves, backend).registradas == 1
    assert _cuarentena(almacen) == []
