use juniper::{
    graphql_object, EmptyMutation, EmptySubscription, GraphQLEnum, GraphQLObject, RootNode,
};

#[derive(GraphQLEnum)]
enum WorkerKind {
    CodeSearch,
}

#[derive(GraphQLObject)]
struct Worker {
    kind: WorkerKind,
    address: String,
}

#[derive(Clone, Copy, Debug)]
pub struct Query;

#[graphql_object]
impl Query {
    fn workers() -> Vec<Worker> {
        vec![]
    }
}

pub type Schema = RootNode<'static, Query, EmptyMutation, EmptySubscription>;

pub fn new() -> Schema {
    Schema::new(Query, EmptyMutation::new(), EmptySubscription::new())
}
