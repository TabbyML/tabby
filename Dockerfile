FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

ENV POETRY_VERSION=1.4.0

WORKDIR /app

RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

RUN pip install "poetry==$POETRY_VERSION"

COPY poetry.lock pyproject.toml /app/
RUN poetry export --without-hashes -o requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

COPY . .
