use std::fs::write;

use juniper::EmptySubscription;
use tabby_webserver::api::{Mutation, Query, Schema};

fn main() {
    let schema = Schema::new(Query, Mutation, EmptySubscription::new());
    write(
        "ee/tabby-webserver/graphql/schema.graphql",
        schema.as_schema_language(),
    )
    .unwrap();
}
