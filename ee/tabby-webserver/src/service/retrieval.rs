use std::{
    collections::HashMap,
    fs::File,
    io::{BufRead, BufReader, Read},
    path::{Path, PathBuf},
    sync::Arc,
};

use tabby_common::api::{
    code::{
        CodeSearch, CodeSearchError, CodeSearchHit, CodeSearchParams, CodeSearchQuery,
        CodeSearchScores,
    },
    structured_doc::{DocSearch, DocSearchError, DocSearchHit},
};
use tabby_schema::{
    context::ContextInfoHelper,
    policy::AccessPolicy,
    repository::{Repository, RepositoryService},
    thread::{CodeQueryInput, CodeSearchParamsOverrideInput, DocQueryInput},
    Result,
};
use tracing::{debug, error, warn};

pub struct RetrievalService {
    code: Arc<dyn CodeSearch>,
    doc: Arc<dyn DocSearch>,
    serper: Option<Box<dyn DocSearch>>,
    repository: Arc<dyn RepositoryService>,
}

impl RetrievalService {
    fn new(
        code: Arc<dyn CodeSearch>,
        doc: Arc<dyn DocSearch>,
        serper: Option<Box<dyn DocSearch>>,
        repository: Arc<dyn RepositoryService>,
    ) -> Self {
        Self {
            code,
            doc,
            serper,
            repository,
        }
    }

    pub async fn collect_file_list(
        &self,
        policy: &AccessPolicy,
        repository: &Repository,
        rev: Option<&str>,
        limit: Option<usize>,
    ) -> Result<(Vec<String>, bool)> {
        match self
            .repository
            .list_files(policy, &repository.kind, &repository.id, rev, limit)
            .await
        {
            Ok((files, truncated)) => {
                let file_list: Vec<_> = files.into_iter().map(|x| x.path).collect();
                Ok((file_list, truncated))
            }
            Err(e) => {
                error!(
                    "failed to list files for repository {}: {}",
                    repository.id, e
                );
                Err(e)
            }
        }
    }

    pub async fn collect_relevant_code(
        &self,
        repository: &Repository,
        helper: &ContextInfoHelper,
        input: &CodeQueryInput,
        params: &CodeSearchParams,
        override_params: Option<&CodeSearchParamsOverrideInput>,
    ) -> Vec<CodeSearchHit> {
        let query = CodeSearchQuery::new(
            input.filepath.clone(),
            input.language.clone(),
            helper.rewrite_tag(&input.content),
            repository.source_id.clone(),
        );

        let mut params = params.clone();
        if let Some(override_params) = override_params {
            override_params.override_params(&mut params);
        }

        match self.code.search_in_language(query, params).await {
            Ok(docs) => merge_code_snippets(repository, docs.hits).await,
            Err(err) => {
                if let CodeSearchError::NotReady = err {
                    debug!("Code search is not ready yet");
                } else {
                    warn!("Failed to search code: {:?}", err);
                }
                vec![]
            }
        }
    }

    pub async fn collect_relevant_docs(
        &self,
        helper: &ContextInfoHelper,
        doc_query: &DocQueryInput,
    ) -> Vec<DocSearchHit> {
        let mut source_ids = doc_query.source_ids.as_deref().unwrap_or_default().to_vec();

        // Only keep source_ids that are valid.
        source_ids.retain(|x| helper.can_access_source_id(x));

        // Rewrite [[source:${id}]] tags to the actual source name for doc search.
        let content = helper.rewrite_tag(&doc_query.content);

        let mut hits = vec![];

        // 1. Collect relevant docs from the tantivy doc search.
        if !source_ids.is_empty() {
            match self.doc.search(&source_ids, &content, 5).await {
                Ok(docs) => hits.extend(docs.hits),
                Err(err) => {
                    if let DocSearchError::NotReady = err {
                        debug!("Doc search is not ready yet");
                    } else {
                        warn!("Failed to search doc: {:?}", err);
                    }
                }
            };
        }

        // 2. If serper is available, we also collect from serper
        if doc_query.search_public {
            if let Some(serper) = self.serper.as_ref() {
                match serper.search(&[], &content, 5).await {
                    Ok(docs) => hits.extend(docs.hits),
                    Err(err) => {
                        warn!("Failed to search serper: {:?}", err);
                    }
                };
            }
        }

        hits
    }
}

pub fn create(
    code: Arc<dyn CodeSearch>,
    doc: Arc<dyn DocSearch>,
    serper: Option<Box<dyn DocSearch>>,
    repository: Arc<dyn RepositoryService>,
) -> RetrievalService {
    RetrievalService::new(code, doc, serper, repository)
}

/// Combine code snippets from search results rather than utilizing multiple hits:
/// Presently, there is only one rule:
/// if the number of lines of code (LoC) is less than 300,
/// and there are multiple hits (number of hits > 1), include the entire file.
pub async fn merge_code_snippets(
    repository: &Repository,
    hits: Vec<CodeSearchHit>,
) -> Vec<CodeSearchHit> {
    // group hits by filepath
    let mut file_hits: HashMap<String, Vec<CodeSearchHit>> = HashMap::new();
    for hit in hits.clone().into_iter() {
        let key = format!("{}-{}", repository.source_id, hit.doc.filepath);
        file_hits.entry(key).or_default().push(hit);
    }

    let mut result = Vec::with_capacity(file_hits.len());

    for (_, file_hits) in file_hits {
        // construct the full path to the file
        let path: PathBuf = repository.dir.join(&file_hits[0].doc.filepath);

        if file_hits.len() > 1 && count_lines(&path).is_ok_and(|x| x < 300) {
            let file_content = read_file_content(&path);

            if let Some(file_content) = file_content {
                debug!(
                    "The file {} is less than 300 lines, so the entire file content will be included",
                    file_hits[0].doc.filepath
                );
                let mut insert_hit = file_hits[0].clone();
                insert_hit.scores =
                    file_hits
                        .iter()
                        .fold(CodeSearchScores::default(), |mut acc, hit| {
                            acc.bm25 += hit.scores.bm25;
                            acc.embedding += hit.scores.embedding;
                            acc.rrf += hit.scores.rrf;
                            acc
                        });
                // average the scores
                let num_files = file_hits.len() as f32;
                insert_hit.scores.bm25 /= num_files;
                insert_hit.scores.embedding /= num_files;
                insert_hit.scores.rrf /= num_files;
                insert_hit.doc.body = file_content;

                // When we use entire file content, mark start_line as None.
                insert_hit.doc.start_line = None;
                result.push(insert_hit);
            }
        } else {
            result.extend(file_hits);
        }
    }

    result.sort_by(|a, b| b.scores.rrf.total_cmp(&a.scores.rrf));
    result
}

/// Read file content and return raw file content string.
pub fn read_file_content(path: &Path) -> Option<String> {
    let mut file = match File::open(path) {
        Ok(file) => file,
        Err(e) => {
            warn!("Error opening file {}: {}", path.display(), e);
            return None;
        }
    };
    let mut content = String::new();
    match file.read_to_string(&mut content) {
        Ok(_) => Some(content),
        Err(e) => {
            warn!("Error reading file {}: {}", path.display(), e);
            None
        }
    }
}

fn count_lines(path: &Path) -> std::io::Result<usize> {
    let mut count = 0;
    for line in BufReader::new(File::open(path)?).lines() {
        line?;
        count += 1;
    }
    Ok(count)
}

#[cfg(test)]
mod tests {
    use std::{path::PathBuf, sync::Arc};

    use juniper::ID;
    use tabby_common::api::{
        code::{CodeSearchDocument, CodeSearchHit, CodeSearchParams, CodeSearchScores},
        structured_doc::{DocSearch, DocSearchDocument},
    };
    use tabby_db::DbConn;
    use tabby_schema::{
        context::{ContextInfo, ContextInfoHelper, ContextSourceValue},
        repository::{Repository, RepositoryKind},
        thread::{CodeQueryInput, CodeSearchParamsOverrideInput, DocQueryInput},
    };

    use super::*;
    use crate::service::{
        access_policy::testutils::make_policy,
        answer::testutils::{make_repository_service, FakeCodeSearch, FakeDocSearch},
    };

    const TEST_SOURCE_ID: &str = "source-1";
    const TEST_GIT_URL: &str = "TabbyML/tabby";
    const TEST_FILEPATH: &str = "test.rs";
    const TEST_LANGUAGE: &str = "rust";
    const TEST_CONTENT: &str = "fn main() {}";

    pub fn make_code_search_params() -> CodeSearchParams {
        CodeSearchParams {
            min_bm25_score: 0.5,
            min_embedding_score: 0.7,
            min_rrf_score: 0.3,
            num_to_return: 5,
            num_to_score: 10,
        }
    }
    pub fn make_code_query_input(source_id: Option<&str>, git_url: Option<&str>) -> CodeQueryInput {
        CodeQueryInput {
            filepath: Some(TEST_FILEPATH.to_string()),
            content: TEST_CONTENT.to_string(),
            git_url: git_url.map(|url| url.to_string()),
            source_id: source_id.map(|id| id.to_string()),
            language: Some(TEST_LANGUAGE.to_string()),
        }
    }

    pub fn make_context_info_helper() -> ContextInfoHelper {
        ContextInfoHelper::new(&ContextInfo {
            sources: vec![ContextSourceValue::Repository(Repository {
                id: ID::from(TEST_SOURCE_ID.to_owned()),
                source_id: TEST_SOURCE_ID.to_owned(),
                name: "tabby".to_owned(),
                kind: RepositoryKind::Github,
                dir: PathBuf::from("tabby"),
                git_url: TEST_GIT_URL.to_owned(),
                refs: vec![],
            })],
        })
    }

    fn get_title(doc: &DocSearchDocument) -> &str {
        match doc {
            DocSearchDocument::Web(web_doc) => &web_doc.title,
            DocSearchDocument::Issue(issue_doc) => &issue_doc.title,
            DocSearchDocument::Pull(pull_doc) => &pull_doc.title,
            DocSearchDocument::Commit(commit_doc) => {
                commit_doc.message.lines().next().unwrap_or(&commit_doc.sha)
            }
        }
    }

    #[tokio::test]
    async fn test_collect_relevant_code() {
        // setup minimal test repository
        let test_repo = Repository {
            id: ID::from("1".to_owned()),
            source_id: TEST_SOURCE_ID.to_owned(),
            name: "test-repo".to_string(),
            kind: RepositoryKind::Git,
            dir: PathBuf::from("test-repo"),
            git_url: TEST_GIT_URL.to_owned(),
            refs: vec![],
        };

        let context_info = ContextInfo {
            sources: vec![ContextSourceValue::Repository(test_repo)],
        };

        let test_repo = Repository {
            id: ID::from("1".to_owned()),
            source_id: TEST_SOURCE_ID.to_owned(),
            name: "test-repo".to_string(),
            kind: RepositoryKind::Git,
            dir: PathBuf::from("test-repo"),
            git_url: TEST_GIT_URL.to_owned(),
            refs: vec![],
        };

        let context_info_helper = ContextInfoHelper::new(&context_info);

        // Setup services
        let code = Arc::new(FakeCodeSearch);
        let doc = Arc::new(FakeDocSearch);
        let db = DbConn::new_in_memory().await.unwrap();
        let repo_service = make_repository_service(db.clone()).await.unwrap();

        let service = RetrievalService::new(code, doc, None, repo_service);

        // Test Case 1: Basic code collection
        let input = make_code_query_input(Some(&test_repo.source_id), Some(&test_repo.git_url));
        let code_hits = service
            .collect_relevant_code(
                &test_repo,
                &context_info_helper,
                &input,
                &make_code_search_params(),
                None,
            )
            .await;
        assert!(!code_hits.is_empty(), "Should find code hits");
        assert!(code_hits[0].scores.rrf > 0.0);

        // Test Case 2: With params override
        let override_params = CodeSearchParamsOverrideInput {
            min_bm25_score: Some(0.1),
            min_embedding_score: Some(0.1),
            min_rrf_score: Some(0.1),
            num_to_return: Some(10),
            num_to_score: Some(20),
        };
        let code_hits_override = service
            .collect_relevant_code(
                &test_repo,
                &context_info_helper,
                &input,
                &make_code_search_params(),
                Some(&override_params),
            )
            .await;
        assert!(
            code_hits_override.len() >= code_hits.len(),
            "Override params should return more hits"
        );
        assert!(
            code_hits_override.iter().all(|hit| hit.scores.rrf >= 0.1),
            "All hits should meet minimum score"
        );
    }

    #[tokio::test]
    async fn test_collect_relevant_docs() {
        let code = Arc::new(FakeCodeSearch);
        let doc = Arc::new(FakeDocSearch);
        let serper = Some(Box::new(FakeDocSearch) as Box<dyn DocSearch>);
        let db = DbConn::new_in_memory().await.unwrap();
        let repo = make_repository_service(db).await.unwrap();

        let service = RetrievalService::new(code.clone(), doc.clone(), serper, repo);

        let context_info_helper = make_context_info_helper();

        // Test Case 1: Test with valid source ID and public search enabled
        let doc_query_1 = DocQueryInput {
            content: "Test query[[source:source-1]]".to_string(),
            source_ids: Some(vec!["source-1".to_string()]),
            search_public: true,
        };

        let hits_1 = service
            .collect_relevant_docs(&context_info_helper, &doc_query_1)
            .await;

        assert_eq!(hits_1.len(), 10);
        assert!(hits_1.iter().any(|hit| get_title(&hit.doc) == "Document 1"));

        // Test Case 2: Test with invalid source ID
        let doc_query_2 = DocQueryInput {
            content: "Test query".to_string(),
            source_ids: Some(vec!["invalid-source".to_string()]),
            search_public: false,
        };

        let hits_2 = service
            .collect_relevant_docs(&context_info_helper, &doc_query_2)
            .await;

        assert_eq!(hits_2.len(), 0);

        // Test Case 3: Test with no source IDs but public search
        let doc_query_3 = DocQueryInput {
            content: "Test query".to_string(),
            source_ids: None,
            search_public: true,
        };

        let hits_3 = service
            .collect_relevant_docs(&context_info_helper, &doc_query_3)
            .await;

        assert!(!hits_3.is_empty());

        // Test Case 4: Test with empty source IDs and no public search
        let doc_query_4 = DocQueryInput {
            content: "Test query".to_string(),
            source_ids: Some(vec![]),
            search_public: false,
        };

        let hits_4 = service
            .collect_relevant_docs(&context_info_helper, &doc_query_4)
            .await;

        assert_eq!(hits_4.len(), 0);
    }

    #[tokio::test]
    async fn test_merge_code_snippets() {
        let db = DbConn::new_in_memory().await.unwrap();
        let repo_service = make_repository_service(db.clone()).await.unwrap();

        let git_url = "https://github.com/test/repo.git".to_string();
        let _id = repo_service
            .git()
            .create("repo".to_string(), git_url.clone())
            .await
            .unwrap();

        let policy = make_policy(db.clone()).await;
        let repo = repo_service
            .repository_list(Some(&policy))
            .await
            .unwrap()
            .pop();

        let hits = vec![
            CodeSearchHit {
                doc: CodeSearchDocument {
                    file_id: "file1".to_string(),
                    chunk_id: "chunk1".to_string(),
                    body: "fn test1() {}\nfn test2() {}".to_string(),
                    filepath: "test.rs".to_string(),
                    git_url: "https://github.com/test/repo.git".to_string(),
                    commit: Some("commit".to_string()),
                    language: "rust".to_string(),
                    start_line: Some(1),
                },
                scores: CodeSearchScores {
                    bm25: 0.5,
                    embedding: 0.7,
                    rrf: 0.3,
                },
            },
            CodeSearchHit {
                doc: CodeSearchDocument {
                    file_id: "file1".to_string(),
                    chunk_id: "chunk2".to_string(),
                    body: "fn test3() {}\nfn test4() {}".to_string(),
                    filepath: "test.rs".to_string(),
                    git_url: "https://github.com/test/repo.git".to_string(),
                    commit: Some("commit".to_string()),
                    language: "rust".to_string(),
                    start_line: Some(3),
                },
                scores: CodeSearchScores {
                    bm25: 0.6,
                    embedding: 0.8,
                    rrf: 0.4,
                },
            },
        ];

        let result = merge_code_snippets(&repo.unwrap(), hits).await;

        assert_eq!(result.len(), 2);
        assert_eq!(result[0].doc.commit, Some("commit".to_string()));
        assert_eq!(result[1].doc.commit, Some("commit".to_string()));
    }
}
