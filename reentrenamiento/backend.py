"""Cliente de las rutas /v1/interno/* del backend (header X-Api-Key)."""

from datetime import datetime
from typing import Any

import httpx


class YaRegistrado(Exception):
    """El backend respondio 409."""


class ClienteBackend:
    def __init__(self, url: str, clave_api: str, cliente: httpx.Client | None = None) -> None:
        self._http = cliente or httpx.Client(base_url=url, timeout=60)
        self._cabeceras = {"X-Api-Key": clave_api}

    def _post(self, ruta: str, cuerpo: dict) -> dict[str, Any]:
        r = self._http.post(ruta, json=cuerpo, headers=self._cabeceras)
        if r.status_code == 409:
            raise YaRegistrado(ruta)
        r.raise_for_status()
        return r.json()

    def registrar_verificada(
        self, paquete_id: str, installation_id: str, blob_path: str, etiqueta: dict
    ) -> str:
        cuerpo = {
            "paquete_id": paquete_id,
            "installation_id": installation_id,
            "blob_path": blob_path,
            "etiqueta_preliminar": etiqueta,
        }
        return self._post("/v1/interno/revision/verificadas", cuerpo)["id"]

    def etiquetas(self, desde: datetime) -> list[dict[str, Any]]:
        r = self._http.get(
            "/v1/interno/revision/etiquetas",
            params={"desde": desde.isoformat()},
            headers=self._cabeceras,
        )
        r.raise_for_status()
        return r.json()

    def marcar_usadas(self, version_modelo: str, ids: list[str]) -> int:
        cuerpo = {"version_modelo": version_modelo, "ids": ids}
        return self._post("/v1/interno/revision/usadas", cuerpo)["marcadas"]

    def registrar_candidato(self, candidato: dict) -> dict[str, Any]:
        return self._post("/v1/interno/modelos/candidatos", candidato)
