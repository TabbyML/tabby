DOCKER_BUILD_ARG:=

build_docker:
	docker buildx build ${DOCKER_BUILD_ARG} -f Dockerfile_base -t tabby_base .
	docker buildx build ${DOCKER_BUILD_ARG} -f Dockerfile -t tabby .
smoke:
	k6 run tests/*.smoke.js

loadtest:
	k6 run tests/*.loadtest.js

fix:
	cargo clippy --fix --allow-dirty --allow-staged && cargo +nightly fmt
