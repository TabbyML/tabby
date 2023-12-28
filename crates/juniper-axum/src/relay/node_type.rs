use std::str::FromStr;

pub trait NodeType {
    /// The [cursor][spec] type that is used for pagination. A cursor
    /// should uniquely identify a given node.
    ///
    /// [spec]: https://relay.dev/graphql/connections.htm#sec-Cursor
    type Cursor: ToString + FromStr + Clone;

    /// Returns the cursor associated with this node.
    fn cursor(&self) -> Self::Cursor;

    /// Returns the type name connections
    /// over these nodes should have in the
    /// API. E.g. `"FooConnection"`.
    fn connection_type_name() -> &'static str;

    /// Returns the type name edges containing
    /// these nodes should have in the API.
    /// E.g. `"FooConnectionEdge"`.
    fn edge_type_name() -> &'static str;
}
