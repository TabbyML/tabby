smoke:
	k6 run tests/*.smoke.js

loadtest:
	k6 run tests/*.loadtest.js
