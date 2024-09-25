fix:
	cargo machete --fix || true
	cargo +nightly fmt
	cargo clippy --fix --allow-dirty --allow-staged

fix-ui:
	pnpm lint:fix

update-ui:
	pnpm build
	rm -rf ee/tabby-webserver/ui && cp -R ee/tabby-ui/out ee/tabby-webserver/ui
	rm -rf ee/tabby-webserver/email_templates && cp -R ee/tabby-email/out ee/tabby-webserver/email_templates

update-db-schema:
	sqlite3 ee/tabby-db/schema.sqlite ".schema --indent" > ee/tabby-db/schema/schema.sql
	sqlite3 ee/tabby-db/schema.sqlite -init  ee/tabby-db/schema/sqlite-schema-visualize.sql "" > schema.dot
	dot -Tsvg schema.dot > ee/tabby-db/schema/schema.svg
	rm schema.dot

dev:
	tmuxinator start -p .tmuxinator/tabby.yml
		
bump-version:
	cargo ws version --force "*" --no-individual-tags --allow-branch "main"

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
	cargo run --package tabby-schema --example update-schema --features=schema-language
