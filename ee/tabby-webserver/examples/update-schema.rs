use std::fs::write;

use juniper::{EmptyMutation, EmptySubscription};
use tabby_webserver::schema::{Query, Schema};

fn main() {
    let schema = Schema::new(Query, EmptyMutation::new(), EmptySubscription::new());
    write(
        "ee/tabby-webserver/graphql/schema.graphql",
        schema.as_schema_language(),
    )
    .unwrap();
}
