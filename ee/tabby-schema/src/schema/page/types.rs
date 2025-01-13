use chrono::{DateTime, Utc};
use juniper::{GraphQLInputObject, GraphQLObject, ID};

use crate::{juniper::relay::NodeType, Context};

#[derive(GraphQLObject)]
#[graphql(context = Context)]
pub struct Page {
    pub id: ID,
    pub author_id: ID,
    pub title: String,
    pub summary: String,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

impl NodeType for Page {
    type Cursor = String;

    fn cursor(&self) -> Self::Cursor {
        self.id.to_string()
    }

    fn connection_type_name() -> &'static str {
        "PageConnection"
    }

    fn edge_type_name() -> &'static str {
        "PageEdge"
    }
}

#[derive(GraphQLObject)]
#[graphql(context = Context)]
pub struct Section {
    pub id: ID,
    pub page_id: ID,
    pub position: i32,
    pub title: String,
    pub content: String,

    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

impl NodeType for Section {
    type Cursor = String;

    fn cursor(&self) -> Self::Cursor {
        self.id.to_string()
    }

    fn connection_type_name() -> &'static str {
        "SectionConnection"
    }

    fn edge_type_name() -> &'static str {
        "SectionEdge"
    }
}

#[derive(GraphQLInputObject)]
pub struct GenerateSectionInput {
    pub page_id: ID,
    pub position: i32,
    pub content: String,
}

#[derive(GraphQLInputObject)]
pub struct ReorderSectionsInput {
    pub page_id: ID,
    pub id: ID,
    pub position: i32,
}
