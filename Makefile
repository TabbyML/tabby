smoke:
	k6 run tests/*.smoke.js

loadtest:
	k6 run tests/*.loadtest.js

fix:
	cargo clippy --fix --allow-dirty --allow-staged && cargo +nightly fmt

update-playground:
	cd clients/tabby-playground && yarn build
	rm -rf crates/tabby/playground && cp -R clients/tabby-playground/out crates/tabby/playground

bump-version:
	cargo ws version --no-git-tag --force "*"

bump-release-version:
	cargo ws version --allow-branch "r*" --no-individual-tags --force "*"
