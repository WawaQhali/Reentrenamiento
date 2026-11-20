"""Configuracion por variables de entorno. Sin secretos: la clave del API interno se lee
de Key Vault con la identidad gestionada (o de INTERNAL_API_KEY en local)."""

import os
from dataclasses import dataclass


def _lista(valor: str) -> tuple[str, ...]:
    return tuple(v.strip() for v in valor.split(",") if v.strip())


@dataclass(frozen=True)
class Ajustes:
    backend_url: str
    storage_account_url: str = ""
    storage_connection_string: str = ""  # solo en local (Azurite)
    keyvault_url: str = ""
    nombre_secreto_api: str = "internal-api-key"
    internal_api_key: str = ""  # solo en local; en Azure se lee de Key Vault

    # Contrato del modelo: tiene que coincidir con el vigente o la compuerta responde 409.
    clases: tuple[str, ...] = ("compatible", "no_compatible")
    clase_positiva: str = "compatible"  # la sensibilidad se mide sobre esta clase
    tamano_entrada: int = 224
    normalizacion: str = "0_255"  # MobileNetV3 de Keras trae su capa de preprocesamiento
    umbral_confianza: float = 0.6

    min_por_clase: int = 10
    epocas: int = 8
    pesos: str = "imagenet"  # "ninguno" en las pruebas, para no descargar nada
    prefijo_datos_base: str = "modelos/datos-base"
    distancia_duplicado: int = 5  # bits distintos del dHash para considerar duplicado
    semilla: int = 7

    @classmethod
    def desde_entorno(cls) -> "Ajustes":
        e = os.environ
        return cls(
            backend_url=e["BACKEND_URL"].rstrip("/"),
            storage_account_url=e.get("STORAGE_ACCOUNT_URL", ""),
            storage_connection_string=e.get("STORAGE_CONNECTION_STRING", ""),
            keyvault_url=e.get("KEYVAULT_URL", ""),
            nombre_secreto_api=e.get("INTERNAL_API_KEY_SECRET", "internal-api-key"),
            internal_api_key=e.get("INTERNAL_API_KEY", ""),
            clases=_lista(e.get("CLASES", "compatible,no_compatible")),
            clase_positiva=e.get("CLASE_POSITIVA", "compatible"),
            tamano_entrada=int(e.get("TAMANO_ENTRADA", "224")),
            normalizacion=e.get("NORMALIZACION", "0_255"),
            umbral_confianza=float(e.get("UMBRAL_CONFIANZA", "0.6")),
            min_por_clase=int(e.get("MIN_POR_CLASE", "10")),
            epocas=int(e.get("EPOCAS", "8")),
            pesos=e.get("PESOS", "imagenet"),
            prefijo_datos_base=e.get("PREFIJO_DATOS_BASE", "modelos/datos-base"),
            distancia_duplicado=int(e.get("DISTANCIA_DUPLICADO", "5")),
        )

    def clave_api(self) -> str:
        if self.internal_api_key:
            return self.internal_api_key
        from azure.identity import DefaultAzureCredential
        from azure.keyvault.secrets import SecretClient

        secreto = SecretClient(self.keyvault_url, DefaultAzureCredential())
        valor = secreto.get_secret(self.nombre_secreto_api).value
        if not valor:
            raise RuntimeError("El secreto del API interno esta vacio")
        return valor
