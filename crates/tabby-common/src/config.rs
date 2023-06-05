use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
#[cfg_attr(feature = "testutils", derive(Serialize))]
pub struct Config {
    pub repositories: Vec<Repository>,
}

impl Config {
    pub fn load() -> Self {
        serdeconv::from_toml_file(crate::path::config_file().as_path())
            .expect("Failed to read config file")
    }

    #[cfg(feature = "testutils")]
    pub fn save(&self) {
        let config_file = crate::path::config_file();
        std::fs::create_dir_all(config_file.parent().unwrap()).unwrap();
        serdeconv::to_toml_file(self, config_file).expect("Failed to write config file")
    }
}

#[derive(Serialize, Deserialize)]
pub struct Repository {
    pub git_url: String,
}
