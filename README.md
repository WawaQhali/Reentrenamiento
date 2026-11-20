# Wawa Qhali — Servicio de reentrenamiento

Servicio aparte del backend que solo usa su API interna (`/v1/interno/*`, header `X-Api-Key`).
Cubre los pasos 4, 6, 7 y 8 del pipeline del informe. Corre en **Azure Container Instances**
por lotes: arranca, procesa y termina.

| Comando | Cuándo | Qué hace |
| --- | --- | --- |
| `verificar` | Cada hora | Lee `cuarentena/`, desenvuelve la clave AES **dentro de Key Vault** (`unwrapKey`, RSA-OAEP-256), descifra con AES-256-GCM (AAD = `installation_id`), exige JPEG de 1024 × 1024, **vuelve a buscar rostros** (OpenCV) y **descarta duplicados** (SHA-256 y dHash). Sube el recorte a `verificadas/` y lo registra en Revisión. El paquete sale de la cuarentena al registrarse o descartarse |
| `entrenar` | Domingos, o a mano | Junta las etiquetas finales del médico (sin `no_evaluable`) y el conjunto base opcional `modelos/datos-base/{clase}/`. Hace ajuste fino de **MobileNetV3Large** (ImageNet) en CPU, mide la **sensibilidad** de `compatible` en validación (20 %), exporta a TFLite y comprueba que no da NaN. Sube `modelos/{versión}/modelo.tflite` y `reporte.json`, registra el **candidato** y marca las imágenes como usadas |

Publicar un candidato **no** lo hace este servicio: lo decide el equipo con
`POST /v1/modelos/{version}/publicar`, y ahí actúa la compuerta de sensibilidad (409 si es menor
que la del vigente).

## Por qué Container Instances y no Container Apps Jobs (ni GPU)

- El entorno de Container Apps de la suscripción de estudiante es «express», y no admite Jobs
  (`ExpressEnvironmentResourceNotSupported`).
- La suscripción no tiene cuota de GPU. Hacer ajuste fino de MobileNetV3 con cientos de imágenes
  en 4 vCPU toma minutos.
- La programación la hace GitHub Actions (`programado.yml`), que arranca el contenedor y espera
  a que termine.

## Seguridad

- **Sin secretos en el repo ni en el contenedor.** La identidad gestionada `id-wawa-reent` lee
  `internal-api-key` de Key Vault al arrancar. Sus permisos son los mínimos:
  - *Key Vault Crypto User* solo sobre la clave `wawa-donaciones`;
  - *Secrets User* solo sobre `internal-api-key`;
  - *Storage Blob Data Contributor*;
  - *AcrPull*.
- La clave privada nunca sale de Key Vault.
- Los logs no llevan el `installation_id` (va como hash) ni datos de la imagen.

## Desarrollo

```bash
python -m uv sync                      # sin TensorFlow
python -m uv run ruff check
python -m uv run pytest                # las 2 pruebas de entrenamiento se omiten
docker build --target pruebas .        # todas, con TensorFlow
```

Variables: `BACKEND_URL`, `STORAGE_ACCOUNT_URL` (o `STORAGE_CONNECTION_STRING` con Azurite),
`KEYVAULT_URL`, `AZURE_CLIENT_ID` (en Azure) o `INTERNAL_API_KEY` (en local). El contrato del
modelo se configura con `CLASES` (por defecto `compatible,no_compatible`), `NORMALIZACION`
(`0_255`), `TAMANO_ENTRADA` (224), `UMBRAL_CONFIANZA` (0.6), `MIN_POR_CLASE` (10) y `EPOCAS` (8).
Tiene que coincidir con el modelo vigente: si cambian las clases o el tamaño de entrada,
`publicar` responde 409.

## Despliegue

CI (`ci.yml`) en cada PR. Al hacer merge a `main`, `cd.yml`:

1. publica `acrwawaqxv8s.azurecr.io/reentrenamiento:<sha>`;
2. recrea `aci-wawa-verificar` (1 vCPU, 2 GiB) y `aci-wawa-entrenar` (4 vCPU, 8 GiB), con la
   identidad `id-wawa-reent`.

Necesita las variables de repo `AZURE_CLIENT_ID` (de `id-wawa-ci`), `AZURE_TENANT_ID` y
`AZURE_SUBSCRIPTION_ID`, las mismas que en Back-End.

A mano: *Actions → Programado → Run workflow* y elegir `verificar` o `entrenar`.
