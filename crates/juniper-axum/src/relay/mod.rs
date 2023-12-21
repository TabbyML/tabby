use std::future::Future;
use juniper::FieldResult;

pub mod connection;
pub mod edge;
pub mod page_info;

pub use connection::Connection;
pub use edge::Edge;
pub use edge::NodeType;
pub use page_info::PageInfo;

pub fn query<Node, F>(
    after: Option<String>,
    before: Option<String>,
    first: Option<i32>,
    last: Option<i32>,
    f: F,
) -> FieldResult<Connection<Node>>
where
    Node: NodeType + Sync,
    F: FnOnce(Option<String>, Option<String>, Option<usize>, Option<usize>) -> FieldResult<Vec<Node>>,
{
    if first.is_some() && last.is_some() {
        return Err("The \"first\" and \"last\" parameters cannot exist at the same time".into());
    }

    let first = match first {
        Some(first) if first < 0 => {
            return Err("The \"first\" parameter must be a non-negative number".into());
        }
        Some(first) => Some(first as usize),
        None => None,
    };

    let last = match last {
        Some(last) if last < 0 => {
            return Err("The \"last\" parameter must be a non-negative number".into());
        }
        Some(last) => Some(last as usize),
        None => None,
    };

    match (first, last) {
        (None, None) => {
            let nodes = f(after, before, None, None)?;
            Ok(Connection::build_connection(nodes, None, None))
        }
        (Some(first), None) => {
            let nodes = f(after, before, Some(first + 1), None)?;
            Ok(Connection::build_connection(nodes, Some(first), None))
        },
        (None, Some(last)) => {
            let nodes = f(after, before, None, Some(last + 1))?;
            Ok(Connection::build_connection(nodes, None, Some(last)))
        },
        _ => Err("The \"first\" and \"last\" parameters cannot exist at the same time".into()),
    }
}

pub async fn query_async<Node, F, R>(
    after: Option<String>,
    before: Option<String>,
    first: Option<i32>,
    last: Option<i32>,
    f: F,
) -> FieldResult<Connection<Node>>
where
    Node: NodeType + Sync,
    F: FnOnce(Option<String>, Option<String>, Option<usize>, Option<usize>) -> R,
    R: Future<Output = FieldResult<Vec<Node>>>,
{
    if first.is_some() && last.is_some() {
        return Err("The \"first\" and \"last\" parameters cannot exist at the same time".into());
    }

    let first = match first {
        Some(first) if first < 0 => {
            return Err("The \"first\" parameter must be a non-negative number".into());
        }
        Some(first) => Some(first as usize),
        None => None,
    };

    let last = match last {
        Some(last) if last < 0 => {
            return Err("The \"last\" parameter must be a non-negative number".into());
        }
        Some(last) => Some(last as usize),
        None => None,
    };

    match (first, last) {
        (None, None) => {
            let nodes = f(after, before, None, None).await?;
            Ok(Connection::build_connection(nodes, None, None))
        }
        (Some(first), None) => {
            let nodes = f(after, before, Some(first + 1), None).await?;
            Ok(Connection::build_connection(nodes, Some(first), None))
        },
        (None, Some(last)) => {
            let nodes = f(after, before, None, Some(last + 1)).await?;
            Ok(Connection::build_connection(nodes, None, Some(last)))
        },
        _ => Err("The \"first\" and \"last\" parameters cannot exist at the same time".into()),
    }
}
