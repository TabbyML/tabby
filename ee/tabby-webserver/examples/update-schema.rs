use std::fs::write;

use tabby_webserver::public::create_schema;

fn main() {
    let schema = create_schema();
    write(
        "ee/tabby-webserver/graphql/schema.graphql",
        schema.as_schema_language(),
    )
    .unwrap();
}
