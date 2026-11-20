import io

import numpy as np
import pytest
from PIL import Image

from reentrenamiento import imagen
from tests.conftest import jpeg, piel_con_ronchas, rostro


def test_jpeg_de_1024_se_acepta() -> None:
    rgb = imagen.abrir_jpeg(jpeg(piel_con_ronchas(1)))
    assert rgb.shape == (1024, 1024, 3)


def test_png_o_tamano_distinto_se_rechazan() -> None:
    png = io.BytesIO()
    Image.fromarray(piel_con_ronchas(1)).save(png, "PNG")
    with pytest.raises(imagen.ImagenRechazada, match="no_es_jpeg"):
        imagen.abrir_jpeg(png.getvalue())
    with pytest.raises(imagen.ImagenRechazada, match="tamano"):
        imagen.abrir_jpeg(jpeg(piel_con_ronchas(1, lado=800)))
    with pytest.raises(imagen.ImagenRechazada, match="jpeg_invalido"):
        imagen.abrir_jpeg(b"no es una imagen")


def test_detecta_un_rostro() -> None:
    assert imagen.hay_rostro(rostro())


@pytest.mark.parametrize("semilla", range(5))
def test_piel_con_ronchas_no_tiene_rostro(semilla: int) -> None:
    assert not imagen.hay_rostro(piel_con_ronchas(semilla))


def test_dhash_reconoce_la_misma_foto_recomprimida_y_distingue_otra() -> None:
    original = piel_con_ronchas(3)
    recomprimida = np.asarray(Image.open(io.BytesIO(jpeg(original, calidad=60))))
    h = imagen.dhash(original)
    assert imagen.distancia(h, imagen.dhash(recomprimida)) <= 5
    assert imagen.distancia(h, imagen.dhash(piel_con_ronchas(4))) > 5
