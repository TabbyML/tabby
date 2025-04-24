pub mod fields {
    pub const KIND: &str = "kind";

    pub mod web {
        pub const TITLE: &str = "title";
        pub const LINK: &str = "link";
        pub const CHUNK_TEXT: &str = "chunk_text";
    }

    pub mod issue {
        pub const TITLE: &str = "title";
        pub const LINK: &str = "link";
        pub const AUTHOR_EMAIL: &str = "author_email";
        pub const BODY: &str = "body";
        pub const CLOSED: &str = "closed";
    }

    pub mod pull {
        pub const TITLE: &str = "title";
        pub const LINK: &str = "link";
        pub const AUTHOR_EMAIL: &str = "author_email";
        pub const BODY: &str = "body";
        pub const DIFF: &str = "diff";
        pub const MERGED: &str = "merged";
    }

    pub mod commit {
        // === Doc level fields ===
        pub const SHA: &str = "sha";
        pub const MESSAGE: &str = "message";
        pub const AUTHOR_EMAIL: &str = "author_email";
        pub const AUTHOR_AT: &str = "author_at";
    }

    pub mod page {
        // === Doc level fields ===
        pub const LINK: &str = "link";
        pub const TITLE: &str = "title";

        // === Chunk level fields ===
        pub const CHUNK_CONTENT: &str = "chunk_text";
    }

    pub mod ingested {
        // === Doc level fields ===
        pub const TITLE: &str = "title";
        pub const LINK: &str = "link";

        // === Chunk level fields ===
        pub const CHUNK_BODY: &str = "chunk_body";
    }
}
