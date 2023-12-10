use clap::crate_version;

pub const CLIENT_ID: &str = concat!(
    env!("CARGO_PKG_REPOSITORY"),
    "/releases/tag/",
    crate_version!()
);
