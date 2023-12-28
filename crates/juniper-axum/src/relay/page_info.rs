use juniper::GraphQLObject;

#[derive(Default, GraphQLObject)]
#[graphql(name = "PageInfo")]
pub struct PageInfo {
    pub has_previous_page: bool,
    pub has_next_page: bool,
    pub start_cursor: Option<String>,
    pub end_cursor: Option<String>,
}
