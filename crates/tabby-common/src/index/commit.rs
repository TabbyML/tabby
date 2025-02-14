pub mod fields {
    // === Doc level fields ===
    pub const GIT_URL: &str = "chunk_git_url";
    pub const SHA: &str = "sha";
    pub const MESSAGE: &str = "message";
    pub const AUTHOR_EMAIL: &str = "author_email";
    pub const AUTHOR_AT: &str = "author_at";
    pub const COMMITTER: &str = "committer";
    pub const COMMIT_AT: &str = "commit_at";

    // === Chunk level fields ===
    pub const CHUNK_FILEPATH: &str = "chunk_filepath";
    pub const CHUNK_DIFF: &str = "chunk_diff";
}
