# syntax = docker/dockerfile:1.5

FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

ARG PYPI_INDEX_URL=https://pypi.org/simple
ARG POETRY_VERSION=1.4.0

WORKDIR /app

RUN --mount=type=cache,target=/root/.cache pip install -i $PYPI_INDEX_URL "poetry==$POETRY_VERSION"

COPY poetry.lock pyproject.toml /app/
RUN poetry export --without-hashes -o requirements.txt

RUN --mount=type=cache,target=/root/.cache pip install -i $PYPI_INDEX_URL -r requirements.txt

COPY . .
