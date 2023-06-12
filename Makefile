smoke:
	k6 run tests/*.smoke.js

loadtest:
	k6 run tests/*.loadtest.js

fix:
	cargo clippy --fix --allow-dirty --allow-staged && cargo +nightly fmt
