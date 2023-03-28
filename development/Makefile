DEFAULT := docker-compose -f ../deployment/docker-compose.yml
UP_FLAGS := up --remove-orphans

build:
	$(DEFAULT) -f docker-compose.dev.yml build

dev:
	$(DEFAULT) -f docker-compose.dev.yml $(UP_FLAGS)

dev-python:
	$(DEFAULT) -f docker-compose.dev.yml -f docker-compose.python.yml $(UP_FLAGS)

clean:
	$(DEFAULT) -f docker-compose.dev.yml down
