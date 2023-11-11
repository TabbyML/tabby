use std::{
    net::SocketAddr,
    time::{SystemTime, UNIX_EPOCH},
};

use anyhow::Result;
use async_trait::async_trait;
use tabby_common::api::code::{CodeSearch, CodeSearchError, SearchResponse};
use tarpc::{client, context, tokio_serde::formats::Json};
use tracing::error;

use crate::CodeSearchServiceClient;

#[derive(Default)]
pub struct CodeSearchWorkerRegistry {
    code: Vec<Box<dyn CodeSearch>>,
}

impl CodeSearchWorkerRegistry {
    pub async fn register(&mut self, server_addr: SocketAddr) -> Result<()> {
        let mut transport = tarpc::serde_transport::tcp::connect(server_addr, Json::default);
        transport.config_mut().max_frame_length(usize::MAX);
        let client =
            CodeSearchServiceClient::new(client::Config::default(), transport.await?).spawn();

        self.code.push(Box::new(client));

        Ok(())
    }
}

#[async_trait]
impl CodeSearch for CodeSearchWorkerRegistry {
    async fn search(
        &self,
        q: &str,
        limit: usize,
        offset: usize,
    ) -> Result<SearchResponse, CodeSearchError> {
        if self.code.is_empty() {
            Err(CodeSearchError::NotReady)
        } else {
            let code = &self.code[random_index(self.code.len())];
            code.search(q, limit, offset).await
        }
    }

    async fn search_in_language(
        &self,
        language: &str,
        tokens: &[String],
        limit: usize,
        offset: usize,
    ) -> Result<SearchResponse, CodeSearchError> {
        if self.code.is_empty() {
            Err(CodeSearchError::NotReady)
        } else {
            let code = &self.code[random_index(self.code.len())];
            code.search_in_language(language, tokens, limit, offset)
                .await
        }
    }
}

fn random_index(size: usize) -> usize {
    let unix_timestamp = (SystemTime::now().duration_since(UNIX_EPOCH))
        .unwrap()
        .as_nanos();
    let index = unix_timestamp % (size as u128);
    index as usize
}

#[async_trait]
impl CodeSearch for CodeSearchServiceClient {
    async fn search(
        &self,
        q: &str,
        limit: usize,
        offset: usize,
    ) -> Result<SearchResponse, CodeSearchError> {
        match self
            .search(context::current(), q.to_owned(), limit, offset)
            .await
        {
            Ok(ret) => Ok(ret),
            Err(err) => {
                error!("RPC error {}", err);
                Err(CodeSearchError::NotReady)
            }
        }
    }

    async fn search_in_language(
        &self,
        language: &str,
        tokens: &[String],
        limit: usize,
        offset: usize,
    ) -> Result<SearchResponse, CodeSearchError> {
        match self
            .search_in_language(
                context::current(),
                language.to_owned(),
                tokens.to_owned(),
                limit,
                offset,
            )
            .await
        {
            Ok(ret) => Ok(ret),
            Err(err) => {
                error!("RPC error {}", err);
                Err(CodeSearchError::NotReady)
            }
        }
    }
}
