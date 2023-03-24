# syntax = docker/dockerfile:1.5

FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

ARG PYPI_INDEX_URL=https://pypi.org/simple
ARG POETRY_VERSION=1.4.0

WORKDIR /app

RUN --mount=type=cache,target=/root/.cache pip install -i $PYPI_INDEX_URL "poetry==$POETRY_VERSION"

COPY poetry.lock pyproject.toml /app/
RUN poetry export --without-hashes -o requirements.txt

RUN --mount=type=cache,target=/root/.cache pip install -i $PYPI_INDEX_URL --extra-index-url https://pypi.org/simple --no-dependencies -r requirements.txt

# Install dagu (https://github.com/yohamta/dagu)
RUN <<EOF
  python -c "import urllib.request; urllib.request.urlretrieve('https://github.com/yohamta/dagu/releases/download/v1.10.2/dagu_1.10.2_Linux_x86_64.tar.gz', 'dagu.tar.gz')"
  tar zxvf dagu.tar.gz
  mv dagu /usr/local/bin
  rm dagu.tar.gz LICENSE.md README.md
EOF

COPY . .
