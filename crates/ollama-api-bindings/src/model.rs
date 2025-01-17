//!
//! Ollama model management utils
//!

use anyhow::{anyhow, bail, Result};
use async_trait::async_trait;
use futures::StreamExt;
use ollama_rs::Ollama;
use tabby_common::config::HttpModelConfig;
use tracing::{info, warn};

/// Env variable for allowing pulling models with Ollama
static ALLOW_PULL_ENV: &str = "TABBY_OLLAMA_ALLOW_PULL";

#[async_trait]
pub trait OllamaModelExt {
    /// Check if a model is available in remote Ollama instance
    async fn model_available(&self, name: impl AsRef<str> + Send) -> Result<bool>;

    /// Get the first available model in remote Ollama instance
    async fn get_first_available_model(&self) -> Result<Option<String>>;

    /// For input model specification:
    /// - If model is specified, check if it is available in remote Ollama instance and returns its name
    /// - If model is not specified and prompt/chat templates are specified, returns a error because it is unsound
    /// - If model is not specified and prompt/chat templates are not specified get the first available model in remote Ollama instance and returns its name
    /// - If no model is available, returns error
    /// - If model is specified and not available, tries to pull it if a env `TABBY_OLLAMA_ALLOW_PULL` equal to `1`, `y`, or `yes`
    ///   and returns error if the environment variable is not set or haves a wrong value
    ///
    /// # Parameters
    /// - `config`: model config configuration
    ///
    /// # Returns
    /// - model name to use
    async fn select_model_or_default(&self, config: &HttpModelConfig) -> Result<String>;

    /// Pull model and puts progress in tracing
    async fn pull_model_with_tracing(&self, model: &str) -> Result<()>;
}

#[async_trait]
impl OllamaModelExt for Ollama {
    async fn model_available(&self, name: impl AsRef<str> + Send) -> Result<bool> {
        let name = name.as_ref();

        let models_available = self.show_model_info(name.into()).await;

        match models_available {
            Ok(_) => Ok(true),
            Err(err) => {
                if err.to_string().contains("not found") {
                    Ok(false)
                } else {
                    Err(err.into())
                }
            }
        }
    }

    async fn get_first_available_model(&self) -> Result<Option<String>> {
        let models_available = self.list_local_models().await?;

        Ok(models_available.first().map(|x| x.name.to_owned()))
    }

    async fn select_model_or_default(&self, config: &HttpModelConfig) -> Result<String> {
        let prompt_or_chat_templates_set =
            config.prompt_template.is_some() || config.chat_template.is_some();

        let model = match config.model_name.to_owned() {
            Some(ref model) => model.to_owned(),
            None => {
                let model = self
                    .get_first_available_model()
                    .await?
                    .ok_or(anyhow!("Ollama instances does not have any models"))?;

                if prompt_or_chat_templates_set {
                    bail!("No model name is provided but prompt or chat templates are set. Please set model name explicitly")
                }

                warn!(
                    "No model name is provided, using first available: {}",
                    model
                );
                model
            }
        };

        let available = self.model_available(&model).await?;

        let allow_pull = std::env::var_os(ALLOW_PULL_ENV)
            .map(|x| x == "1" || x.eq_ignore_ascii_case("y") || x.eq_ignore_ascii_case("yes"))
            .unwrap_or(false);

        match (available, allow_pull) {
            (true, _) => Ok(model),
            (false, true) => {
                info!("Model is not available, pulling it");
                self.pull_model_with_tracing(model.as_str()).await?;
                Ok(model)
            }
            (false, false) => {
                bail!("Model is not available, and pulling is disabled")
            }
        }
    }

    async fn pull_model_with_tracing(&self, model: &str) -> Result<()> {
        let mut stream = self.pull_model_stream(model.to_owned(), false).await?;

        let mut last_status = "".to_string();
        let mut last_progress = 0.0;

        while let Some(result) = stream.next().await {
            let response = result?;
            let status = response.message;
            if last_status != status {
                info!("Status: {}", status);
                last_status = status;
                last_progress = 0.0;
            }

            // Show progress only if 1% gain happened
            if let (Some(completed), Some(total)) = (response.completed, response.total) {
                let progress = completed as f64 / total as f64;
                if progress - last_progress > 0.01 {
                    info!("Progress: {:.2}%", progress * 100.0);
                    last_progress = progress;
                }
            }
        }

        Ok(())
    }
}
