UP_FLAGS := up --remove-orphans --remove-orphans
DEV_FLAGS := $(UP_FLAGS) --build

up:
	docker-compose -f docker-compose.yml $(UP_FLAGS)

up-triton:
	docker-compose -f docker-compose.yml -f docker-compose.triton.yml $(UP_FLAGS)

dev:
	docker-compose -f docker-compose.yml -f docker-compose.dev.yml $(DEV_FLAGS)

dev-triton:
	docker-compose -f docker-compose.yml -f docker-compose.triton.yml -f docker-compose.dev.yml $(DEV_FLAGS)
