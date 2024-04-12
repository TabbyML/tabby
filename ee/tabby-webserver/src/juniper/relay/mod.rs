use std::future::Future;

use juniper::FieldResult;

mod connection;
mod edge;
mod node_type;
mod page_info;

pub use connection::Connection;

pub use node_type::NodeType;


fn validate_first_last(
    first: Option<i32>,
    last: Option<i32>,
) -> FieldResult<(Option<usize>, Option<usize>)> {
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
    Ok((first, last))
}

#[allow(unused)]
pub fn query<Node, F>(
    after: Option<String>,
    before: Option<String>,
    first: Option<i32>,
    last: Option<i32>,
    f: F,
) -> FieldResult<Connection<Node>>
where
    Node: NodeType + Sync,
    F: FnOnce(
        Option<String>,
        Option<String>,
        Option<usize>,
        Option<usize>,
    ) -> FieldResult<Vec<Node>>,
{
    let (first, last) = validate_first_last(first, last)?;

    let first_plus_one = first.map(|i| i + 1);
    let last_plus_one = last.map(|i| i + 1);
    let after_some = after.is_some();
    let before_some = before.is_some();
    let nodes = f(after, before, first_plus_one, last_plus_one)?;
    Ok(Connection::build_connection(
        nodes,
        after_some,
        before_some,
        first,
        last,
    ))
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
    let (first, last) = validate_first_last(first, last)?;

    let first_plus_one = first.map(|i| i + 1);
    let last_plus_one = last.map(|i| i + 1);
    let after_some = after.is_some();
    let before_some = before.is_some();
    let nodes = f(after, before, first_plus_one, last_plus_one).await?;
    Ok(Connection::build_connection(
        nodes,
        after_some,
        before_some,
        first,
        last,
    ))
}
