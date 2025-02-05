use chrono::{DateTime, Utc};
use juniper::{GraphQLInputObject, GraphQLObject, GraphQLUnion, ID};

use crate::{juniper::relay::NodeType, Context};

#[derive(GraphQLObject)]
#[graphql(context = Context)]
pub struct Page {
    pub id: ID,
    pub author_id: ID,
    pub title: Option<String>,
    pub content: Option<String>,
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
pub struct AddPageSectionInput {
    pub page_id: ID,
    pub title: String,
}

#[derive(GraphQLObject)]
pub struct PageCreated {
    pub id: ID,
    pub author_id: ID,
}

#[derive(GraphQLObject)]
pub struct PageTitleDelta {
    pub delta: String,
}

#[derive(GraphQLObject)]
pub struct PageTitleCompleted {
    pub id: ID,
}

#[derive(GraphQLObject)]
pub struct PageContentDelta {
    pub delta: String,
}

#[derive(GraphQLObject)]
pub struct PageContentCompleted {
    pub id: ID,
}

#[derive(GraphQLObject)]
pub struct PageSectionCreated {
    pub id: ID,
    pub title: String,
    pub content: String,
}

/// Schema of page convert stream.
#[derive(GraphQLUnion)]
#[graphql(context = Context)]
pub enum PageConvertItem {
    PageCreated(PageCreated),
    PageTitleDelta(PageTitleDelta),
    PageTitleCompleted(PageTitleCompleted),
    PageContentDelta(PageContentDelta),
    PageContentCompleted(PageContentCompleted),

    // PageSection is converted from thread messages,
    // will return title and content directly instead of delta.
    // At least one PageSectionCreated will be present in the stream.
    PageSectionCreated(PageSectionCreated),
}
