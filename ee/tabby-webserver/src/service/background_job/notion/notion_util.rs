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

pub async fn fetch_all_pages(access_token: &str, database_id: &str) -> Result<Vec<PageWithBlocks>, anyhow::Error> {
    let client = Client::new(access_token.to_string(), None)?;
    let pages = fetch_all_pages_without_blocks(&client, database_id).await?;
    let mut results = Vec::new();
    for page in pages {
        let blocks = fetch_nested_blocks(&client, page.id.as_str()).await?;
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
        results.push(PageWithBlocks{page, block_map,children});
    }
    Ok(results)
}





  