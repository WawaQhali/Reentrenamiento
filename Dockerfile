# syntax=docker/dockerfile:1

# --- Etapa 1: dependencias con uv (incluye TensorFlow para CPU) ----------------
FROM python:3.12-slim AS build
COPY --from=ghcr.io/astral-sh/uv:0.13.8 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /srv
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --group entrenamiento --no-install-project
COPY reentrenamiento ./reentrenamiento

# --- Pruebas (docker build --target pruebas .): las de entrenamiento necesitan TensorFlow
FROM build AS pruebas
RUN uv sync --frozen --group entrenamiento --no-install-project
COPY tests ./tests
RUN uv run --no-sync pytest -q

# --- Imagen final, sin uv y con usuario no root ----------------------------------
FROM python:3.12-slim
RUN useradd --create-home --uid 10001 wawa
WORKDIR /srv
COPY --from=build --chown=wawa:wawa /srv /srv
ENV PATH="/srv/.venv/bin:$PATH" PYTHONUNBUFFERED=1 TF_CPP_MIN_LOG_LEVEL=2
USER wawa
# Pesos de ImageNet dentro de la imagen: el entrenamiento no depende de descargas.
RUN python -c "import tensorflow as tf; tf.keras.applications.MobileNetV3Large(input_shape=(224, 224, 3), include_top=False, weights='imagenet')"
ENTRYPOINT ["python", "-m", "reentrenamiento"]
CMD ["verificar"]
