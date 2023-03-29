POETRY_EXISTS := $(shell which poetry &> /dev/null)
PRE_COMMIT_EXISTS := $(shell poetry run which pre-commit &> /dev/null)
PRE_COMMIT_HOOK := .git/hooks/pre-commit

pre-commit: setup-development-environment
	poetry run pre-commit

install-poetry:
ifndef POETRY_EXISTS
	curl -sSL https://install.python-poetry.org | POETRY_VERSION=1.4.0 python3 -
endif

ifndef PRE_COMMIT_EXISTS
	poetry run pip install $(shell poetry export --without-hashes --with dev | grep pre-commit | cut -d";" -f1)
endif

$(PRE_COMMIT_HOOK):
	poetry run pre-commit install --install-hooks

setup-development-environment: install-poetry $(PRE_COMMIT_HOOK)
