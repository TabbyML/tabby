use juniper::{
    macros::reflect::{BaseSubTypes, BaseType, Type, Types, WrappedType, WrappedValue},
    marker::IsOutputType,
    meta::MetaType,
    Arguments, ExecutionResult, Executor, GraphQLType, GraphQLValue, GraphQLValueAsync, Registry,
    ScalarValue,
};

use super::{edge::Edge, page_info::PageInfo, NodeType};

/// Connection type
///
/// Connection is the result of a query for `relay::query` or `relay::query_async`
pub struct Connection<Node> {
    /// All edges of the current page.
    pub edges: Vec<Edge<Node>>,
    pub page_info: PageInfo,
}

impl<S, T: WrappedType<S>> WrappedType<S> for Connection<T> {
    const VALUE: WrappedValue = T::VALUE * 10 + 3;
}

impl<S, T: BaseType<S>> BaseType<S> for Connection<T> {
    const NAME: Type = T::NAME;
}

impl<S, T: BaseSubTypes<S>> BaseSubTypes<S> for Connection<T> {
    const NAMES: Types = T::NAMES;
}

impl<Node> Connection<Node>
where
    Node: NodeType,
{
    #[allow(unused)]
    /// Returns a relay relay with no elements.
    pub fn empty() -> Self {
        Self {
            edges: Vec::new(),
            page_info: PageInfo::default(),
        }
    }

    pub fn build_connection(
        nodes: Vec<Node>,
        after: bool,
        before: bool,
        first: Option<usize>,
        last: Option<usize>,
    ) -> Self {
        let selected_count = first.or(last).unwrap_or(nodes.len());
        let len = nodes.len();
        let has_next_page = if last.is_some() {
            before
        } else {
            selected_count < len
        };
        let has_previous_page = if last.is_some() {
            selected_count < len
        } else {
            after
        };

        let edges: Vec<_> = if let Some(last) = last {
            nodes
                .into_iter()
                .rev()
                .take(last)
                .rev()
                .map(|node| {
                    let cursor = node.cursor();
                    Edge::new(cursor.to_string(), node)
                })
                .collect()
        } else {
            nodes
                .into_iter()
                .take(first.unwrap_or(len))
                .map(|node| {
                    let cursor = node.cursor();
                    Edge::new(cursor.to_string(), node)
                })
                .collect()
        };

        Connection {
            page_info: PageInfo {
                has_previous_page,
                has_next_page,
                start_cursor: edges.first().map(|edge| edge.cursor.clone()),
                end_cursor: edges.last().map(|edge| edge.cursor.clone()),
            },
            edges,
        }
    }
}

impl<Node, S> GraphQLType<S> for Connection<Node>
where
    Node: NodeType + GraphQLType<S>,
    Node::Context: juniper::Context,
    S: ScalarValue,
{
    fn name(_info: &Self::TypeInfo) -> Option<&str> {
        Some(Node::connection_type_name())
    }

    fn meta<'r>(info: &Self::TypeInfo, registry: &mut Registry<'r, S>) -> MetaType<'r, S>
    where
        S: 'r,
    {
        let fields = [
            registry.field::<&Vec<Edge<Node>>>("edges", info),
            registry.field::<&PageInfo>("pageInfo", &()),
        ];
        registry
            .build_object_type::<Self>(info, &fields)
            .into_meta()
    }
}

impl<Node, S> GraphQLValue<S> for Connection<Node>
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
            "edges" => executor.resolve_with_ctx(info, &self.edges),
            "pageInfo" => executor.resolve_with_ctx(&(), &self.page_info),
            _ => panic!("Field {field_name} not found on type ConnectionEdge"),
        }
    }

    fn concrete_type_name(&self, _context: &Self::Context, info: &Self::TypeInfo) -> String {
        self.type_name(info).unwrap_or("Connection").to_string()
    }
}

impl<Node, S> GraphQLValueAsync<S> for Connection<Node>
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
    ) -> juniper::BoxFuture<'a, ExecutionResult<S>> {
        let f = async move {
            match field_name {
                "edges" => executor.resolve_with_ctx_async(info, &self.edges).await,
                "pageInfo" => executor.resolve_with_ctx(&(), &self.page_info),
                _ => panic!("Field {field_name} not found on type ConnectionEdge"),
            }
        };
        use ::juniper::futures::future;
        future::FutureExt::boxed(f)
    }
}

impl<Node, S> IsOutputType<S> for Connection<Node>
where
    Node: GraphQLType<S> + NodeType,
    Node::Context: juniper::Context,
    S: ScalarValue,
{
}
