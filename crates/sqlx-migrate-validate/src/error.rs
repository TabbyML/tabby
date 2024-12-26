use sqlx::migrate::MigrateError;

#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum ValidateError {
    #[error("migration {0} was not applied")]
    VersionNotApplied(i64),

    #[error(transparent)]
    MigrateError(#[from] MigrateError),
}
