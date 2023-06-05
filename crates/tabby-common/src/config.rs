use serde::Deserialize;

#[derive(Deserialize)]
pub struct Config {
    pub repositories: Vec<Repository>,
}

impl Config {
    pub fn load() -> Self {
        serdeconv::from_toml_file(crate::path::CONFIG_FILE.as_path())
            .expect("Failed to read config file")
    }
}

#[derive(Deserialize)]
pub struct Repository {
    pub git_url: String,
}
