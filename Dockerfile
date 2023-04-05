# syntax = docker/dockerfile:1.5

FROM tabbyml/fastertransformer_backend

RUN apt update && apt -y install build-essential libssl-dev zlib1g-dev \
  libbz2-dev libreadline-dev libsqlite3-dev curl \
  libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev

RUN mkdir -p /home/app
RUN chown 1000 /home/app

USER 1000
WORKDIR /home/app
ENV HOME /home/app

# Setup pyenv
RUN git clone --depth=1 https://github.com/pyenv/pyenv.git .pyenv
ENV PATH="$HOME/.pyenv/shims:/home/app/.pyenv/bin:$PATH"

ARG PYTHON_VERSION=3.10.10
RUN pyenv install ${PYTHON_VERSION}
RUN pyenv global ${PYTHON_VERSION}

ARG PYPI_INDEX_URL=https://pypi.org/simple
ARG POETRY_VERSION=1.4.0

RUN --mount=type=cache,target=$HOME/.cache pip install -i $PYPI_INDEX_URL "poetry==$POETRY_VERSION"

# vector
RUN <<EOF
curl --proto '=https' --tlsv1.2 -sSf https://sh.vector.dev | bash -s -- -y
EOF
ENV PATH "$HOME/.vector/bin:$PATH"

# Supervisord
RUN --mount=type=cache,target=$HOME/.cache pip install -i $PYPI_INDEX_URL supervisor

RUN mkdir -p ~/.bin
ENV PATH "$HOME/.bin:$PATH"

# Install dagu
RUN <<EOF
  curl -L https://github.com/yohamta/dagu/releases/download/v1.10.2/dagu_1.10.2_Linux_x86_64.tar.gz > dagu.tar.gz
  tar zxvf dagu.tar.gz
  mv dagu ~/.bin/
  rm dagu.tar.gz LICENSE.md README.md
EOF

# Install tabby dependencies
COPY poetry.lock pyproject.toml ./
RUN poetry export --without-hashes > requirements.txt
RUN --mount=type=cache,target=$HOME/.cache pip install -i $PYPI_INDEX_URL --no-dependencies -r requirements.txt


COPY tabby ./tabby

# Setup file permissions
USER root
RUN mkdir -p /var/lib/vector
RUN chown 1000 /var/lib/vector

RUN mkdir -p $HOME/.cache
RUN chown 1000 $HOME/.cache

USER 1000
CMD ["./tabby/scripts/tabby.sh"]
