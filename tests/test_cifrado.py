import base64
import uuid

import pytest

from reentrenamiento.cifrado import PaqueteInvalido, descifrar
from tests.conftest import ETIQUETA


def _contenido() -> dict:
    return {
        "imagen_jpeg": base64.b64encode(b"\xff\xd8jpeg\xff\xd9").decode(),
        "etiqueta_preliminar": ETIQUETA,
    }


def test_ida_y_vuelta_con_el_formato_de_la_app(claves) -> None:  # noqa: ANN001
    instalacion = uuid.uuid4()
    paquete = claves.cifrar_para_pruebas(str(instalacion), _contenido())
    assert descifrar(paquete, instalacion, claves) == _contenido()


def test_con_otro_installation_id_falla_por_el_aad(claves) -> None:  # noqa: ANN001
    paquete = claves.cifrar_para_pruebas(str(uuid.uuid4()), _contenido())
    with pytest.raises(PaqueteInvalido) as error:
        descifrar(paquete, uuid.uuid4(), claves)
    assert error.value.motivo == "aad_o_datos_alterados"


@pytest.mark.parametrize(
    ("cambio", "motivo"),
    [
        ({"v": 2}, "version"),
        ({"nonce": base64.b64encode(b"corto").decode()}, "tamanos"),
        ({"datos": "no es base64 !!"}, "base64"),
        ({"kid": "https://otro.vault/keys/x/1"}, "kid_desconocido"),
    ],
)
def test_paquetes_mal_formados_se_rechazan(claves, cambio: dict, motivo: str) -> None:  # noqa: ANN001
    instalacion = uuid.uuid4()
    paquete = {**claves.cifrar_para_pruebas(str(instalacion), _contenido()), **cambio}
    with pytest.raises(PaqueteInvalido) as error:
        descifrar(paquete, instalacion, claves)
    assert error.value.motivo == motivo
