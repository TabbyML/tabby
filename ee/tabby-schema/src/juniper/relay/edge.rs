use juniper::{
    marker::IsOutputType, meta::MetaType, Arguments, BoxFuture, ExecutionResult, Executor,
    GraphQLType, GraphQLValue, GraphQLValueAsync, Registry, ScalarValue,
};

use super::NodeType;

/// An edge in a relay.
pub struct Edge<Node> {
    pub cursor: String,
    pub node: Node,
}

impl<Node> Edge<Node> {
    /// Create a new edge.
    #[inline]
    pub fn new(cursor: String, node: Node) -> Self {
        Self { cursor, node }
    }
}

impl<Node, S> GraphQLType<S> for Edge<Node>
where
    Node: NodeType + GraphQLType<S>,
    Node::Context: juniper::Context,
    S: ScalarValue,
{
    fn name(_info: &Self::TypeInfo) -> Option<&str> {
        Some(Node::edge_type_name())
    }

    fn meta<'r>(info: &Self::TypeInfo, registry: &mut Registry<'r, S>) -> MetaType<'r, S>
    where
        S: 'r,
    {
        let fields = [
            registry.field::<&Node>("node", info),
            registry.field::<&String>("cursor", &()),
        ];
        registry
            .build_object_type::<Self>(info, &fields)
            .into_meta()
    }
}

impl<Node, S> GraphQLValue<S> for Edge<Node>
where
    Node: NodeType + GraphQLType<S>,
    Node::Context: juniper::Context,
    S: ScalarValue,
{
    type Context = Node::Context;
    type TypeInfo = <Node as GraphQLValue<S>>::TypeInfo;

    fn type_name<'i>(&self, info: &'i Self::TypeInfo) -> Option<&'i str> {
        <Self as GraphQLType<S>>::name(info)
    }

    fn resolve_field(
        &self,
        info: &Self::TypeInfo,
        field_name: &str,
        _arguments: &Arguments<S>,
        executor: &Executor<Self::Context, S>,
    ) -> ExecutionResult<S> {
        match field_name {
            "node" => executor.resolve_with_ctx(info, &self.node),
            "cursor" => executor.resolve_with_ctx(&(), &self.cursor),
            _ => panic!("Field {field_name} not found on type ConnectionEdge"),
        }
    }

    fn concrete_type_name(&self, _context: &Self::Context, info: &Self::TypeInfo) -> String {
        self.type_name(info).unwrap_or("ConnectionEdge").to_string()
    }
}

impl<Node, S> GraphQLValueAsync<S> for Edge<Node>
where
    Node: NodeType + GraphQLType<S> + GraphQLValueAsync<S> + Send + Sync,
    Node::TypeInfo: Sync,
    Node::Context: juniper::Context + Sync,
    S: ScalarValue + Send + Sync,
{
    fn resolve_field_async<'a>(
        &'a self,
        info: &'a Self::TypeInfo,
        field_name: &'a str,
        _arguments: &'a Arguments<S>,
        executor: &'a Executor<Self::Context, S>,
    ) -> BoxFuture<'a, ExecutionResult<S>> {
        let f = async move {
            match field_name {
                "node" => executor.resolve_with_ctx_async(info, &self.node).await,
                "cursor" => executor.resolve_with_ctx(&(), &self.cursor),
                _ => panic!("Field {field_name} not found on type RelayConnectionEdge"),
            }
        };
        use ::juniper::futures::future;
        future::FutureExt::boxed(f)
    }
}

impl<Node, S> IsOutputType<S> for Edge<Node>
where
    Node: GraphQLType<S> + NodeType,
    Node::Context: juniper::Context,
    S: ScalarValue,
{
}
