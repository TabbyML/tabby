use chrono::{DateTime, Utc};
use juniper::{GraphQLEnum, GraphQLObject, GraphQLUnion, ID};

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
    pub position: i32,

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

#[derive(GraphQLEnum)]
pub enum MoveSectionDirection {
    Up,
    Down,
}

#[derive(GraphQLObject)]
pub struct PageCreated {
    pub id: ID,
    pub author_id: ID,
    pub title: String,
}

#[derive(GraphQLObject)]
pub struct PageSectionsCreated {
    pub sections: Vec<PageSection>,
}

#[derive(GraphQLObject, Clone)]
pub struct PageSection {
    pub id: ID,
    pub title: String,
    pub position: i32,
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
pub struct PageSectionContentDelta {
    pub id: ID,
    pub delta: String,
}

#[derive(GraphQLObject)]
pub struct PageSectionContentCompleted {
    pub id: ID,
}

#[derive(GraphQLObject)]
pub struct PageCompleted {
    pub id: ID,
}

/// Schema of page convert stream.
#[derive(GraphQLUnion)]
#[graphql(context = Context)]
pub enum PageRunItem {
    // PageCreated will return at the beginning of the stream,
    // containing the page ID, author and title.
    PageCreated(PageCreated),

    PageContentDelta(PageContentDelta),
    PageContentCompleted(PageContentCompleted),

    // PageSectionsCreated will return the titles of all sections.
    PageSectionsCreated(PageSectionsCreated),

    PageSectionContentDelta(PageSectionContentDelta),
    PageSectionContentCompleted(PageSectionContentCompleted),

    PageCompleted(PageCompleted),
}
