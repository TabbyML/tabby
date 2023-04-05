DEFAULT :=
UP_FLAGS := up --remove-orphans

build:
	docker-compose -f ../deployment/docker-compose.yml -f docker-compose.dev.yml build

dev:
	docker-compose -f ../deployment/docker-compose.yml -f docker-compose.dev.yml $(UP_FLAGS)

dev-python:
	docker-compose -f ../deployment/docker-compose.python.yml -f docker-compose.dev.yml $(UP_FLAGS)
