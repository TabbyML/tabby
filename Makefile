POETRY_EXISTS := $(shell which poetry &> /dev/null)
PRE_COMMIT_HOOK := .git/hooks/pre-commit

pre-commit:
	poetry run pre-commit

install-poetry:
ifndef POETRY_EXISTS
	curl -sSL https://install.python-poetry.org | POETRY_VERSION=1.4.0 python3 -
endif
	poetry install

$(PRE_COMMIT_HOOK):
	poetry run pre-commit install --install-hooks

setup-development-environment: install-poetry $(PRE_COMMIT_HOOK)
