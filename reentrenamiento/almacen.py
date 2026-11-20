"""Blob Storage. Las rutas son `contenedor/nombre`, igual que en el backend."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


def separar(ruta: str) -> tuple[str, str]:
    contenedor, _, nombre = ruta.partition("/")
    if not contenedor or not nombre:
        raise ValueError(f"Ruta de blob invalida: {ruta}")
    return contenedor, nombre


@dataclass(frozen=True)
class Blob:
    ruta: str
    metadatos: dict[str, str] = field(default_factory=dict)


class Almacen(ABC):
    @abstractmethod
    def listar(self, prefijo: str) -> list[Blob]:
        """Blobs cuya ruta empieza por `prefijo` (que incluye el contenedor)."""

    @abstractmethod
    def leer(self, ruta: str) -> bytes: ...

    @abstractmethod
    def subir(self, ruta: str, datos: bytes, tipo: str, metadatos: dict[str, str]) -> None:
        """Sobrescribe si ya existe: reintentar con el mismo paquete no debe fallar."""

    @abstractmethod
    def borrar(self, ruta: str) -> None:
        """No falla si el blob ya no existe."""


class AlmacenMemoria(Almacen):
    def __init__(self) -> None:
        self.blobs: dict[str, tuple[bytes, dict[str, str]]] = {}

    def listar(self, prefijo: str) -> list[Blob]:
        return [
            Blob(r, dict(m)) for r, (_, m) in sorted(self.blobs.items()) if r.startswith(prefijo)
        ]

    def leer(self, ruta: str) -> bytes:
        return self.blobs[ruta][0]

    def subir(self, ruta: str, datos: bytes, tipo: str, metadatos: dict[str, str]) -> None:
        self.blobs[ruta] = (datos, dict(metadatos))

    def borrar(self, ruta: str) -> None:
        self.blobs.pop(ruta, None)


class AlmacenAzure(Almacen):
    """Con identidad gestionada; en local, con la cadena de conexion de Azurite."""

    def __init__(self, account_url: str, connection_string: str = "") -> None:
        from azure.storage.blob import BlobServiceClient

        if connection_string:
            self._cliente = BlobServiceClient.from_connection_string(connection_string)
        else:
            from azure.identity import DefaultAzureCredential

            self._cliente = BlobServiceClient(account_url, credential=DefaultAzureCredential())

    def listar(self, prefijo: str) -> list[Blob]:
        contenedor, _, resto = prefijo.partition("/")
        cliente = self._cliente.get_container_client(contenedor)
        return [
            Blob(f"{contenedor}/{b.name}", dict(b.metadata or {}))
            for b in cliente.list_blobs(name_starts_with=resto or None, include=["metadata"])
        ]

    def leer(self, ruta: str) -> bytes:
        contenedor, nombre = separar(ruta)
        return self._cliente.get_blob_client(contenedor, nombre).download_blob().readall()

    def subir(self, ruta: str, datos: bytes, tipo: str, metadatos: dict[str, str]) -> None:
        from azure.storage.blob import ContentSettings

        contenedor, nombre = separar(ruta)
        self._cliente.get_blob_client(contenedor, nombre).upload_blob(
            datos,
            overwrite=True,
            metadata=metadatos,
            content_settings=ContentSettings(content_type=tipo),
        )

    def borrar(self, ruta: str) -> None:
        from azure.core.exceptions import ResourceNotFoundError

        contenedor, nombre = separar(ruta)
        try:
            self._cliente.get_blob_client(contenedor, nombre).delete_blob()
        except ResourceNotFoundError:
            pass
