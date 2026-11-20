"""Descifrado de las donaciones (seccion 6.4 de la guia del backend).

La clave AES se desenvuelve DENTRO de Key Vault (`unwrapKey`, RSA-OAEP-256): la clave privada
nunca llega a este servicio. El `installation_id` es el dato adicional autenticado (AAD).
"""

import base64
import binascii
import json
import uuid
from abc import ABC, abstractmethod
from typing import Any

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.hazmat.primitives.ciphers.aead import AESGCM


class PaqueteInvalido(Exception):
    """El paquete no tiene el formato esperado o no se puede descifrar. `motivo` va al log."""

    def __init__(self, motivo: str) -> None:
        super().__init__(motivo)
        self.motivo = motivo


class Desenvolvedor(ABC):
    @abstractmethod
    def desenvolver(self, kid: str, clave_cifrada: bytes) -> bytes:
        """Devuelve la clave AES de 32 bytes."""


class DesenvolvedorKeyVault(Desenvolvedor):
    def __init__(self) -> None:
        from azure.identity import DefaultAzureCredential

        self._credencial = DefaultAzureCredential()
        self._clientes: dict[str, Any] = {}

    def desenvolver(self, kid: str, clave_cifrada: bytes) -> bytes:
        from azure.keyvault.keys.crypto import CryptographyClient, KeyWrapAlgorithm

        if kid not in self._clientes:
            self._clientes[kid] = CryptographyClient(kid, self._credencial)
        return self._clientes[kid].unwrap_key(KeyWrapAlgorithm.rsa_oaep_256, clave_cifrada).key


class DesenvolvedorLocal(Desenvolvedor):
    """Para las pruebas: una clave RSA de 3072 bits en memoria."""

    OAEP = padding.OAEP(
        mgf=padding.MGF1(algorithm=hashes.SHA256()), algorithm=hashes.SHA256(), label=None
    )

    def __init__(self) -> None:
        self.privada = rsa.generate_private_key(public_exponent=65537, key_size=3072)
        self.kid = "https://local.vault/keys/wawa-donaciones/prueba"

    def desenvolver(self, kid: str, clave_cifrada: bytes) -> bytes:
        if kid != self.kid:
            raise PaqueteInvalido("kid_desconocido")
        return self.privada.decrypt(clave_cifrada, self.OAEP)

    def cifrar_para_pruebas(self, installation_id: str, contenido: dict) -> dict:
        """Lo mismo que hace la app (seccion 6.2)."""
        import os

        clave, nonce = os.urandom(32), os.urandom(12)
        datos = AESGCM(clave).encrypt(
            nonce, json.dumps(contenido).encode(), installation_id.encode()
        )
        envuelta = self.privada.public_key().encrypt(clave, self.OAEP)
        b64 = base64.b64encode
        return {
            "v": 1,
            "kid": self.kid,
            "alg": "RSA-OAEP-256+A256GCM",
            "clave_cifrada": b64(envuelta).decode(),
            "nonce": b64(nonce).decode(),
            "datos": b64(datos).decode(),
        }


def _b64(paquete: dict, campo: str) -> bytes:
    try:
        return base64.b64decode(paquete[campo], validate=True)
    except (KeyError, TypeError, binascii.Error) as error:
        raise PaqueteInvalido("base64") from error


def descifrar(paquete: dict, installation_id: uuid.UUID, desenvolvedor: Desenvolvedor) -> dict:
    """Devuelve el JSON en claro: `imagen_jpeg` (base64) y `etiqueta_preliminar`."""
    if paquete.get("v") != 1 or paquete.get("alg") != "RSA-OAEP-256+A256GCM":
        raise PaqueteInvalido("version")
    clave_cifrada, nonce, datos = (_b64(paquete, c) for c in ("clave_cifrada", "nonce", "datos"))
    if len(clave_cifrada) != 384 or len(nonce) != 12 or len(datos) < 16:
        raise PaqueteInvalido("tamanos")

    clave = desenvolvedor.desenvolver(str(paquete.get("kid", "")), clave_cifrada)
    try:
        claro = AESGCM(clave).decrypt(nonce, datos, str(installation_id).encode())
    except InvalidTag as error:
        raise PaqueteInvalido("aad_o_datos_alterados") from error
    finally:
        del clave

    try:
        contenido = json.loads(claro)
    except ValueError as error:
        raise PaqueteInvalido("json") from error
    if not isinstance(contenido, dict) or not isinstance(contenido.get("imagen_jpeg"), str):
        raise PaqueteInvalido("contenido")
    if not isinstance(contenido.get("etiqueta_preliminar"), dict):
        raise PaqueteInvalido("contenido")
    return contenido
