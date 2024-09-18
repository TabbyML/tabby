use notion_client::endpoints::{
    databases::query::request::QueryDatabaseRequest,
    Client,
};
use chrono::{DateTime, NaiveDate, Utc};
use notion_client::objects::parent;
use notion_client::{
    objects::page::Page,
    objects::page::PageProperty,
    objects::block::Block,
};
use std::collections::HashMap;

pub struct PageWithBlocks {
    pub page: Page,
    pub block_map: HashMap::<String, Block>,
    pub children: HashMap::<String, Vec<String>>,
}





impl PageWithBlocks {
    // page_properties returns the properties of the page in a string format.
    // The properties are separated by a newline character.
    // Currently, only the title and rich_text properties are supported.
    fn page_properties(&self) -> String {
        let mut res = Vec::new();
        for (k, v) in self.page.properties.iter() {
            match v {
                PageProperty::Title { title, .. } => {
                    res.push(k.clone());
                    let s = title.iter().map(|t| t.plain_text()).filter_map(|text| text).collect::<Vec<String>>().join(" ");
                    res.push(s);
                },
                PageProperty::RichText { rich_text, .. } => {
                    res.push(k.clone());
                    let s = rich_text.iter().map(|t| t.plain_text()).filter_map(|text| text).collect::<Vec<String>>().join(" ");
                    res.push(s);
                },
                _ => {
                    
                }
            }
        }
        return res.join("\n");
    }
    
    // dfs returns content in depth-first order.
    fn dfs(&self) ->String {
       
        let mut results = vec![];
        let mut stack = vec![(0, self.page.id.clone())];
        while let Some((cur_indent, cur_id)) = stack.pop() {
            if let Some(block) = self.block_map.get(&cur_id){
                results.push(format!("{}{}", " ".repeat(cur_indent), block.block_type.plain_text().into_iter().filter_map(|f| f).collect::<Vec<String>>().join(" ")));
            }
            if let Some(children_ids) = self.children.get(&cur_id) {
                for child_id in children_ids.iter().rev() {
                    stack.push((cur_indent+1, child_id.to_string()));
                }
            }
        }
        return results.join("\n");
    }
    // plain_text returns the properties and contents of the page in a string format.
    // The content is separated by a newline character.
    pub fn plain_text(&self) -> String {
        let mut res = vec![];
        res.push(self.page_properties());
        res.push("Content".to_string());
        res.push(self.dfs());
        return res.join("\n");
        
    }
    pub fn title(&self) -> String {
        self.page.properties.iter().find_map( |(_, v)| {
            match v {
                PageProperty::Title { title, .. } => {
                    let s = title.iter().map(|t| t.plain_text()).filter_map(|text| text).collect::<Vec<String>>().join(" ");
                    Some(s)
                },
                _ => None
            }
        }).unwrap_or("".to_string())
    }
    
    pub fn id(&self) -> String {
        self.page.id.clone()
    }

    pub fn url(&self) -> String {
        self.page.url.clone()
    }

    pub fn last_edited_time(&self) -> DateTime<Utc> {
        self.page.last_edited_time.clone()
    }

}

// fetch_all_pages fetches all pages in a database using notion api WITHOUT BLOCKS
async fn fetch_all_pages_without_blocks(client: &Client, database_id: &str) ->Result<Vec<Page>,anyhow::Error> {
    let mut has_more = true;
    let mut next_cursor = None;
    let mut results = Vec::new();

    while has_more{
        let res = client.databases.query_a_database(database_id, QueryDatabaseRequest{start_cursor:next_cursor.clone(), filter: None, sorts: None, page_size: None }).await?; 
        if res.results.is_empty() {
            break;
        }
        has_more = res.has_more;
        next_cursor = res.next_cursor;
        results.extend(res.results);
    }
    Ok(results)
} 

// fetch_all_blocks fetches all blocks in a page using notion api WITHOUT NESTED CHILDREN.
async fn fetch_all_blocks(client: &Client, page_id:&str) -> Result<Vec<Block>,anyhow::Error> {
    let mut results = Vec::new();
    let mut has_more = true;
    let mut next_cursor:Option<String> = None;
    while has_more {
        let res: notion_client::endpoints::blocks::retrieve::response::RetrieveBlockChilerenResponse =  client.blocks.retrieve_block_children(
            page_id,
            next_cursor.as_deref(),
            Some(32))
            .await?;
        if res.results.is_empty() {
            break;
        }
        has_more = res.has_more;
        next_cursor = res.next_cursor;
        results.extend(res.results);
    }
    Ok(results)
}

// fetch_nested_blocks fetches all blocks in a page using notion api WITH NESTED CHILDREN.
pub async fn fetch_nested_blocks(client: &Client, page_id:&str) -> Result<Vec<Block>,anyhow::Error> {
    let mut stack = vec![page_id.to_string()];
    let mut results = Vec::new();
    while let Some(page_id) = stack.pop() {
        let blocks = fetch_all_blocks(client, page_id.as_str()).await?;
        for block in blocks.iter().filter(|b: &&Block| b.has_children == Some(true)) {
            if let Some(id) = &block.id {
                stack.push(id.to_string());
            }
        }
        results.extend(blocks);
    }
    Ok(results)

}

fn construct_page(page:Page, blocks: Vec<Block>) -> PageWithBlocks {
    let mut children = HashMap::<String, Vec<String>>::new();
    let mut block_map = HashMap::<String, Block>::new();
    for block in blocks {
        if let Some(id) = &block.id {
            if let  Some(parent) = block.parent.as_ref() {
                match parent {
                    parent::Parent::PageId { page_id } => children.entry(page_id.to_string()).or_insert_with(Vec::new).push(id.to_string()),
                    parent::Parent::BlockId { block_id } => children.entry(block_id.to_string()).or_insert_with(Vec::new).push(id.to_string()),
                    _ => (),
                }
            }
            block_map.insert(id.to_string(), block);
        }
    }
    PageWithBlocks{page, block_map,children}
}

pub async fn fetch_all_pages(access_token: &str, database_id: &str) -> Result<Vec<PageWithBlocks>, anyhow::Error> {
    let client = Client::new(access_token.to_string(), None)?;
    let pages = fetch_all_pages_without_blocks(&client, database_id).await?;
    let mut results = Vec::new();
    for page in pages {
        let blocks = fetch_nested_blocks(&client, page.id.as_str()).await?;
        results.push(construct_page(page, blocks));
    }
    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::construct_page;
    use notion_client::objects::{block::{self, Block, BlockType}, page::{Page, PageProperty}, parent::Parent, rich_text::RichText, user::User};

    #[tokio::test]
    async fn test_page_plain_text() {
        let page_str = r#"{"archived":false,"created_by":{"id":"6bf32dea-f316-4f40-8dc4-90363e224020","object":"user"},"created_time":"2024-09-01T08:58:00Z","icon":{"external":{"url":"https://www.notion.so/icons/clipping_lightgray.svg"},"type":"external"},"id":"402031b0-7a3f-4400-b4a8-b3bc0e909a2d","last_edited_by":{"id":"6bf32dea-f316-4f40-8dc4-90363e224020","object":"user"},"last_edited_time":"2024-09-01T08:58:00Z","parent":{"database_id":"bf9e48ab-bf7e-45db-85e3-307768ed94de","type":"database_id"},"properties":{"Assignee":{"id":"notion%3A%2F%2Ftasks%2Fassign_property","people":[{"id":"ac7a3bd0-c111-4464-8f45-8a857a1abc8a","object":"user"}],"type":"people"},"Due":{"date":{"end":null,"start":"2023-05-28","time_zone":null},"id":"notion%3A%2F%2Ftasks%2Fdue_date_property","type":"date"},"GitHub Pull Requests":{"has_more":false,"id":"notion%3A%2F%2Ftasks%2Ftask_to_github_prs_relation","relation":[],"type":"relation"},"Parent-task":{"has_more":false,"id":"notion%3A%2F%2Ftasks%2Fparent_task_relation","relation":[],"type":"relation"},"Priority":{"id":"notion%3A%2F%2Ftasks%2Fpriority_property","select":{"color":"green","id":"priority_low","name":"Low"},"type":"select"},"Project":{"has_more":false,"id":"notion%3A%2F%2Ftasks%2Ftask_to_project_relation","relation":[{"id":"adc8d1ea-e174-432a-bc83-059120ad5a32"}],"type":"relation"},"Sprint":{"has_more":false,"id":"notion%3A%2F%2Ftasks%2Ftask_sprint_relation","relation":[],"type":"relation"},"Status":{"id":"notion%3A%2F%2Ftasks%2Fstatus_property","status":{"color":"default","id":"not-started","name":"Not Started"},"type":"status"},"Sub-tasks":{"has_more":false,"id":"notion%3A%2F%2Ftasks%2Fsub_task_relation","relation":[],"type":"relation"},"Summary":{"id":"notion%3A%2F%2Ftasks%2Fai_summary_property","rich_text":[{"annotations":{"bold":false,"code":false,"color":"default","italic":false,"strikethrough":false,"underline":false},"plain_text":"This document outlines the scope of the tooltip project, including goals of uncovering technical blockers and estimating engineering time, as well as non-goals of evaluating new events and collecting feedback. The next steps include revisiting docs from a previous project and writing a tech spec to share with the product team email alias.","text":{"content":"This document outlines the scope of the tooltip project, including goals of uncovering technical blockers and estimating engineering time, as well as non-goals of evaluating new events and collecting feedback. The next steps include revisiting docs from a previous project and writing a tech spec to share with the product team email alias."},"type":"text"}],"type":"rich_text"},"Tags":{"id":"notion%3A%2F%2Ftasks%2Ftags_property","multi_select":[{"color":"purple","id":"Mobile","name":"Mobile"},{"color":"blue","id":"Website","name":"Website"}],"type":"multi_select"},"Task ID":{"id":"notion%3A%2F%2Ftasks%2Fauto_increment_id_property","type":"unique_id","unique_id":{"number":5,"prefix":null}},"Task name":{"id":"title","title":[{"annotations":{"bold":false,"code":false,"color":"default","italic":false,"strikethrough":false,"underline":false},"plain_text":"Scope tooltip project","text":{"content":"Scope tooltip project"},"type":"text"}],"type":"title"}},"url":"https://www.notion.so/Scope-tooltip-project-402031b07a3f4400b4a8b3bc0e909a2d"}"#;
        let page: Page = serde_json::from_str(page_str).unwrap();
        let mut blocks = vec![];
        let blocks_str = r#"{"archived":false,"bulleted_list_item":{"color":"default","rich_text":[{"annotations":{"bold":false,"code":false,"color":"default","italic":false,"strikethrough":false,"underline":false},"plain_text":"Goals:","text":{"content":"Goals:"},"type":"text"}]},"created_by":{"id":"6bf32dea-f316-4f40-8dc4-90363e224020","object":"user"},"created_time":"2024-09-01T08:58:00Z","has_children":true,"id":"8c6b5284-3e43-4632-a97e-810c3a0f0a52","last_edited_by":{"id":"6bf32dea-f316-4f40-8dc4-90363e224020","object":"user"},"last_edited_time":"2024-09-01T08:58:00Z","object":"block","parent":{"page_id":"402031b0-7a3f-4400-b4a8-b3bc0e909a2d","type":"page_id"},"type":"bulleted_list_item"}
{"archived":false,"bulleted_list_item":{"color":"default","rich_text":[{"annotations":{"bold":false,"code":false,"color":"default","italic":false,"strikethrough":false,"underline":false},"plain_text":"Non-goals:","text":{"content":"Non-goals:"},"type":"text"}]},"created_by":{"id":"6bf32dea-f316-4f40-8dc4-90363e224020","object":"user"},"created_time":"2024-09-01T08:58:00Z","has_children":true,"id":"258e8d11-02fb-4726-9a4a-9c456c30a69f","last_edited_by":{"id":"6bf32dea-f316-4f40-8dc4-90363e224020","object":"user"},"last_edited_time":"2024-09-01T08:58:00Z","object":"block","parent":{"page_id":"402031b0-7a3f-4400-b4a8-b3bc0e909a2d","type":"page_id"},"type":"bulleted_list_item"}
{"archived":false,"bulleted_list_item":{"color":"default","rich_text":[{"annotations":{"bold":false,"code":false,"color":"default","italic":false,"strikethrough":false,"underline":false},"plain_text":"Next steps:","text":{"content":"Next steps:"},"type":"text"}]},"created_by":{"id":"6bf32dea-f316-4f40-8dc4-90363e224020","object":"user"},"created_time":"2024-09-01T08:58:00Z","has_children":true,"id":"3887737f-511c-4cc1-98ee-d582a7ea5789","last_edited_by":{"id":"6bf32dea-f316-4f40-8dc4-90363e224020","object":"user"},"last_edited_time":"2024-09-01T08:58:00Z","object":"block","parent":{"page_id":"402031b0-7a3f-4400-b4a8-b3bc0e909a2d","type":"page_id"},"type":"bulleted_list_item"}
{"archived":false,"bulleted_list_item":{"color":"default","rich_text":[{"annotations":{"bold":false,"code":false,"color":"default","italic":false,"strikethrough":false,"underline":false},"plain_text":"Revisit docs from previous tooltip project from Q1","text":{"content":"Revisit docs from previous tooltip project from Q1"},"type":"text"}]},"created_by":{"id":"6bf32dea-f316-4f40-8dc4-90363e224020","object":"user"},"created_time":"2024-09-01T08:58:00Z","has_children":false,"id":"fe8a9d70-1de2-4c84-91eb-a5b5c5a3ca7e","last_edited_by":{"id":"6bf32dea-f316-4f40-8dc4-90363e224020","object":"user"},"last_edited_time":"2024-09-01T08:58:00Z","object":"block","parent":{"block_id":"3887737f-511c-4cc1-98ee-d582a7ea5789","type":"block_id"},"type":"bulleted_list_item"}
{"archived":false,"bulleted_list_item":{"color":"default","rich_text":[{"annotations":{"bold":false,"code":false,"color":"default","italic":false,"strikethrough":false,"underline":false},"plain_text":"Write tech spec and share to product team email alias","text":{"content":"Write tech spec and share to product team email alias"},"type":"text"}]},"created_by":{"id":"6bf32dea-f316-4f40-8dc4-90363e224020","object":"user"},"created_time":"2024-09-01T08:58:00Z","has_children":false,"id":"fe071346-ce4c-41a4-98f4-6d9854f0df61","last_edited_by":{"id":"6bf32dea-f316-4f40-8dc4-90363e224020","object":"user"},"last_edited_time":"2024-09-01T08:58:00Z","object":"block","parent":{"block_id":"3887737f-511c-4cc1-98ee-d582a7ea5789","type":"block_id"},"type":"bulleted_list_item"}
{"archived":false,"bulleted_list_item":{"color":"default","rich_text":[{"annotations":{"bold":false,"code":false,"color":"default","italic":false,"strikethrough":false,"underline":false},"plain_text":"Evaluate new events for data team to track","text":{"content":"Evaluate new events for data team to track"},"type":"text"}]},"created_by":{"id":"6bf32dea-f316-4f40-8dc4-90363e224020","object":"user"},"created_time":"2024-09-01T08:58:00Z","has_children":false,"id":"3a8d107c-4cb3-49dd-a424-586f72f31ccf","last_edited_by":{"id":"6bf32dea-f316-4f40-8dc4-90363e224020","object":"user"},"last_edited_time":"2024-09-01T08:58:00Z","object":"block","parent":{"block_id":"258e8d11-02fb-4726-9a4a-9c456c30a69f","type":"block_id"},"type":"bulleted_list_item"}
{"archived":false,"bulleted_list_item":{"color":"default","rich_text":[{"annotations":{"bold":false,"code":false,"color":"default","italic":false,"strikethrough":false,"underline":false},"plain_text":"Collect feedback from cross-functional partners","text":{"content":"Collect feedback from cross-functional partners"},"type":"text"}]},"created_by":{"id":"6bf32dea-f316-4f40-8dc4-90363e224020","object":"user"},"created_time":"2024-09-01T08:58:00Z","has_children":false,"id":"02d40641-1ef4-4409-a30c-e82ca6f15ad0","last_edited_by":{"id":"6bf32dea-f316-4f40-8dc4-90363e224020","object":"user"},"last_edited_time":"2024-09-01T08:58:00Z","object":"block","parent":{"block_id":"258e8d11-02fb-4726-9a4a-9c456c30a69f","type":"block_id"},"type":"bulleted_list_item"}
{"archived":false,"bulleted_list_item":{"color":"default","rich_text":[{"annotations":{"bold":false,"code":false,"color":"default","italic":false,"strikethrough":false,"underline":false},"plain_text":"Uncover any technical blockers to building education tooltips","text":{"content":"Uncover any technical blockers to building education tooltips"},"type":"text"}]},"created_by":{"id":"6bf32dea-f316-4f40-8dc4-90363e224020","object":"user"},"created_time":"2024-09-01T08:58:00Z","has_children":false,"id":"ee26f843-fd2d-4614-bad9-8fb4a1260a81","last_edited_by":{"id":"6bf32dea-f316-4f40-8dc4-90363e224020","object":"user"},"last_edited_time":"2024-09-01T08:58:00Z","object":"block","parent":{"block_id":"8c6b5284-3e43-4632-a97e-810c3a0f0a52","type":"block_id"},"type":"bulleted_list_item"}
{"archived":false,"bulleted_list_item":{"color":"default","rich_text":[{"annotations":{"bold":false,"code":false,"color":"default","italic":false,"strikethrough":false,"underline":false},"plain_text":"Estimate how long engineering will need to build tooltips","text":{"content":"Estimate how long engineering will need to build tooltips"},"type":"text"}]},"created_by":{"id":"6bf32dea-f316-4f40-8dc4-90363e224020","object":"user"},"created_time":"2024-09-01T08:58:00Z","has_children":false,"id":"58f06acb-516e-4e51-bccd-9d373d1f738a","last_edited_by":{"id":"6bf32dea-f316-4f40-8dc4-90363e224020","object":"user"},"last_edited_time":"2024-09-01T08:58:00Z","object":"block","parent":{"block_id":"8c6b5284-3e43-4632-a97e-810c3a0f0a52","type":"block_id"},"type":"bulleted_list_item"}"#;
        for block_str in blocks_str.split("\n") {
            blocks.push(serde_json::from_str(block_str).unwrap());
        }
        let p = construct_page(page, blocks);
        println!("{}", p.plain_text());
        let result = "Task name
Scope tooltip project
Summary
This document outlines the scope of the tooltip project, including goals of uncovering technical blockers and estimating engineering time, as well as non-goals of evaluating new events and collecting feedback. The next steps include revisiting docs from a previous project and writing a tech spec to share with the product team email alias.
Content
 Goals:
  Uncover any technical blockers to building education tooltips
  Estimate how long engineering will need to build tooltips
 Non-goals:
  Evaluate new events for data team to track
  Collect feedback from cross-functional partners
 Next steps:
  Revisit docs from previous tooltip project from Q1
  Write tech spec and share to product team email alias";
        assert_eq!(p.plain_text(), result);

    }
}