UP_FLAGS := up --remove-orphans --remove-orphans --build

dev:
	docker-compose -f docker-compose.yml -f docker-compose.dev.yml $(UP_FLAGS)

dev-triton:
	docker-compose -f docker-compose.yml -f docker-compose.triton.yml -f docker-compose.dev.yml $(UP_FLAGS)

clean:
	docker-compose -f docker-compose.yml -f docker-compose.triton.yml -f docker-compose.dev.yml down
