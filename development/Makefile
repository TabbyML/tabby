DEFAULT :=
UP_FLAGS := up --remove-orphans

build:
	docker-compose -f ../deployment/docker-compose.yml -f docker-compose.dev.yml build

dev:
	docker-compose -f ../deployment/docker-compose.yml -f docker-compose.dev.yml $(UP_FLAGS)

dev-triton:
	docker-compose -f ../deployment/docker-compose.yml -f ../deployment/docker-compose.triton.yml -f docker-compose.dev.yml $(UP_FLAGS)
