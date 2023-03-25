POETRY_EXISTS := $(shell which poetry &> /dev/null)
PRE_COMMIT_HOOK := .git/hooks/pre-commit
LOCAL_MODEL := testdata/tiny-70M/models/fastertransformer/1

pre-commit:
	poetry run pre-commit

install-poetry:
ifndef POETRY_EXISTS
	curl -sSL https://install.python-poetry.org | POETRY_VERSION=1.4.0 python3 -
endif
	poetry install

$(PRE_COMMIT_HOOK):
	poetry run pre-commit install --install-hooks

$(LOCAL_MODEL):
	poetry run python -m tabby.tools.huggingface_gptneox_convert \
		-in_file EleutherAI/pythia-70m-deduped \
		-o $@ \
		-i_g 1 -m_n tiny-70M -p 1 -w fp16

setup-development-environment: install-poetry $(PRE_COMMIT_HOOK)


UP_FLAGS := up --remove-orphans --remove-orphans
DEV_FLAGS := $(UP_FLAGS) --build

up:
	docker-compose -f deployment/docker-compose.yml $(UP_FLAGS)

up-triton: $(LOCAL_MODEL)
	docker-compose -f deployment/docker-compose.yml -f deployment/docker-compose.triton.yml $(UP_FLAGS)

dev:
	docker-compose -f deployment/docker-compose.yml -f deployment/docker-compose.dev.yml $(DEV_FLAGS)

dev-triton: $(LOCAL_MODEL)
	docker-compose -f deployment/docker-compose.yml -f deployment/docker-compose.triton.yml -f deployment/docker-compose.dev.yml $(DEV_FLAGS)
