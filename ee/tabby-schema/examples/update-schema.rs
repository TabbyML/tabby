use std::fs::write;

use tabby_schema::create_schema;

fn main() {
    let schema = create_schema();
    write("ee/tabby-schema/graphql/schema.graphql", schema.as_sdl()).unwrap();
}
