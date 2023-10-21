loadtest:
ifdef TABBY_API_HOST
	k6 run tests/*.loadtest.js
else 
	$(error TABBY_API_HOST is undefined)
endif

fix:
	cargo clippy --fix --allow-dirty --allow-staged && cargo +nightly fmt

update-playground:
	cd clients/tabby-playground && yarn build
	rm -rf crates/tabby/playground && cp -R clients/tabby-playground/out crates/tabby/playground

bump-version:
	cargo ws version --no-git-tag --force "*"

bump-release-version:
	cargo ws version --allow-branch "r*" --no-individual-tags --force "*"

update-openapi-doc:
	curl http://localhost:8080/api-docs/openapi.json | jq '                                                       \
	  delpaths([                                                                                                  \
		  ["paths", "/v1beta/chat/completions"],                                                                    \
		  ["paths", "/v1beta/search"],                                                                              \
		  ["components", "schemas", "CompletionRequest", "properties", "debug_options"],                            \
		  ["components", "schemas", "CompletionResponse", "properties", "debug_data"],                              \
		  ["components", "schemas", "DebugData"],                                                                   \
			["components", "schemas", "DebugOptions"]                                                                 \
			])' | jq '.servers[0] |= { url: "https://playground.app.tabbyml.com", description: "Playground server" }' \
			    > website/static/openapi.json
