use thiserror::Error;

#[derive(Error, Debug)]
pub enum CompletionError {
    #[error("failed to format prompt")]
    MiniJinja(#[from] minijinja::Error),
}
