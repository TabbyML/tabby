use std::ops::Range;

use async_trait::async_trait;
use juniper::{GraphQLObject, ID};
use juniper_axum::relay::NodeType;
use tabby_common::SourceFile;
use validator::Validate;

use super::{Context, Result};

#[derive(Validate)]
pub struct CreateRepositoryInput {
    #[validate(regex(
        code = "name",
        path = "tabby_common::config::REPOSITORY_NAME_REGEX",
        message = "Invalid repository name"
    ))]
    pub name: String,
    #[validate(url(code = "gitUrl", message = "Invalid Git URL"))]
    pub git_url: String,
}

#[derive(GraphQLObject, Debug)]
#[graphql(context = Context)]
pub struct Repository {
    pub id: juniper::ID,
    pub name: String,
    pub git_url: String,
}

#[derive(GraphQLObject, Debug)]
pub struct FileEntry {
    pub r#type: String,
    pub path: String,
}

#[derive(GraphQLObject)]
pub struct RepositoryMeta {
    pub git_url: String,
    pub filepath: String,
    pub language: String,
    pub max_line_length: i32,
    pub avg_line_length: f64,
    pub alphanum_fraction: f64,
    pub tags: Vec<Tag>,
}

impl From<SourceFile> for RepositoryMeta {
    fn from(value: SourceFile) -> Self {
        RepositoryMeta {
            git_url: value.git_url,
            filepath: value.filepath,
            language: value.language,
            max_line_length: value.max_line_length as i32,
            avg_line_length: value.avg_line_length as f64,
            alphanum_fraction: value.alphanum_fraction as f64,
            tags: value
                .tags
                .into_iter()
                .map(|tag| Tag {
                    range: tag.range.into(),
                    name_range: tag.name_range.into(),
                    utf16_column_range: tag.utf16_column_range.into(),
                    span: tag.span.into(),
                    line_range: tag.line_range.into(),
                    docs: tag.docs,
                    is_definition: tag.is_definition,
                    syntax_type_name: tag.syntax_type_name,
                })
                .collect(),
        }
    }
}

#[derive(GraphQLObject)]
pub struct Tag {
    pub range: IntRange,
    pub name_range: IntRange,
    pub utf16_column_range: IntRange,
    pub span: PointRange,
    pub line_range: IntRange,
    pub docs: Option<String>,
    pub is_definition: bool,
    pub syntax_type_name: String,
}

#[derive(GraphQLObject)]
pub struct Point {
    pub row: i32,
    pub col: i32,
}

impl From<Range<tabby_common::Point>> for PointRange {
    fn from(value: Range<tabby_common::Point>) -> Self {
        PointRange {
            start: Point {
                row: value.start.row as i32,
                col: value.end.column as i32,
            },
            end: Point {
                row: value.end.row as i32,
                col: value.end.column as i32,
            },
        }
    }
}

#[derive(GraphQLObject)]
pub struct IntRange {
    pub start: i32,
    pub end: i32,
}

impl From<Range<usize>> for IntRange {
    fn from(value: Range<usize>) -> Self {
        IntRange {
            start: value.start as i32,
            end: value.end as i32,
        }
    }
}

#[derive(GraphQLObject)]
pub struct PointRange {
    pub start: Point,
    pub end: Point,
}

impl NodeType for Repository {
    type Cursor = String;

    fn cursor(&self) -> Self::Cursor {
        self.id.to_string()
    }

    fn connection_type_name() -> &'static str {
        "RepositoryConnection"
    }

    fn edge_type_name() -> &'static str {
        "RepositoryEdge"
    }
}

#[async_trait]
pub trait RepositoryService: Send + Sync {
    async fn list_repositories(
        &self,
        after: Option<String>,
        before: Option<String>,
        first: Option<usize>,
        last: Option<usize>,
    ) -> Result<Vec<Repository>>;

    async fn create_repository(&self, name: String, git_url: String) -> Result<ID>;
    async fn delete_repository(&self, id: &ID) -> Result<bool>;
    async fn update_repository(&self, id: &ID, name: String, git_url: String) -> Result<bool>;

    async fn search_files(
        &self,
        name: String,
        path_glob: String,
        top_n: usize,
    ) -> Result<Vec<FileEntry>>;
    async fn repository_meta(&self, name: String, path: String) -> Result<RepositoryMeta>;
}
