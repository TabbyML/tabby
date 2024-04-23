fix:
	cargo machete --fix || true
	cargo +nightly fmt
	cargo clippy --fix --allow-dirty --allow-staged

fix-ui:
	cd ee/tabby-ui && yarn format:write && yarn lint:fix

update-ui:
	cd ee/tabby-ui && yarn build
	rm -rf ee/tabby-webserver/ui && cp -R ee/tabby-ui/out ee/tabby-webserver/ui

update-email-templates:
	cd ee/tabby-email && yarn export
	rm -rf ee/tabby-webserver/email_templates && cp -R ee/tabby-email/out ee/tabby-webserver/email_templates

caddy:
	caddy run --watch --config ee/tabby-webserver/development/Caddyfile

bump-version:
	cargo ws version --no-git-tag --force "*"

bump-release-version:
	cargo ws version --allow-branch "r*" --no-individual-tags --force "*"

update-openapi-doc:
	curl http://localhost:8080/api-docs/openapi.json | jq '                                                       \
	delpaths([                                                                                                    \
		  ["paths", "/v1beta/chat/completions"],                                                                  \
		  ["paths", "/v1beta/search"],                                                                            \
		  ["paths", "/v1beta/server_setting"],                                                                    \
		  ["components", "schemas", "CompletionRequest", "properties", "prompt"],                                 \
		  ["components", "schemas", "CompletionRequest", "properties", "debug_options"],                          \
		  ["components", "schemas", "CompletionResponse", "properties", "debug_data"],                            \
		  ["components", "schemas", "DebugData"],                                                                 \
		  ["components", "schemas", "DebugOptions"],                                                              \
		  ["components", "schemas", "ServerSetting"]                                                              \
	  ])' | jq '.servers[0] |= { url: "https://playground.app.tabbyml.com", description: "Playground server" }'   \
			    > website/static/openapi.json

update-graphql-schema:
	cargo run --package tabby-webserver --example update-schema
