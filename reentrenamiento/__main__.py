"""Uso: python -m reentrenamiento verificar | entrenar | todo"""

import logging
import sys

from reentrenamiento.almacen import AlmacenAzure
from reentrenamiento.backend import ClienteBackend
from reentrenamiento.config import Ajustes

COMANDOS = ("verificar", "entrenar", "todo")


def principal(argv: list[str]) -> int:
    if len(argv) != 1 or argv[0] not in COMANDOS:
        print(__doc__, file=sys.stderr)
        return 2
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    for ruidoso in ("azure", "httpx", "urllib3"):
        logging.getLogger(ruidoso).setLevel(logging.WARNING)
    log = logging.getLogger("reentrenamiento")

    ajustes = Ajustes.desde_entorno()
    almacen = AlmacenAzure(ajustes.storage_account_url, ajustes.storage_connection_string)
    backend = ClienteBackend(ajustes.backend_url, ajustes.clave_api())

    if argv[0] in ("verificar", "todo"):
        from reentrenamiento import verificar
        from reentrenamiento.cifrado import DesenvolvedorKeyVault

        resumen = verificar.ejecutar(
            almacen, DesenvolvedorKeyVault(), backend, ajustes.distancia_duplicado
        )
        log.info("Verificacion: %s", resumen)

    if argv[0] in ("entrenar", "todo"):
        from reentrenamiento import entrenar

        reporte = entrenar.ejecutar(ajustes, almacen, backend)
        log.info("Entrenamiento: %s", "sin candidato" if reporte is None else reporte["version"])
    return 0


if __name__ == "__main__":
    sys.exit(principal(sys.argv[1:]))
