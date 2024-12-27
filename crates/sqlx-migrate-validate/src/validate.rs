use std::{
    borrow::Cow,
    collections::{HashMap, HashSet},
};

use async_trait::async_trait;
use sqlx::migrate::{
    AppliedMigration, Migrate, MigrateError, Migration, MigrationSource, Migrator,
};

use crate::error::ValidateError;

#[async_trait(?Send)]
pub trait Validate {
    /// Validate previously applied migrations against the migration source.
    /// Depending on the migration source this can be used to check if all migrations
    /// for the current version of the application have been applied.
    /// Use [`Validator::from_migrator`] to use the migrations available during compilation.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// # sqlx_rt::block_on(async move {
    /// # use sqlx_migrate_validate::Validator;
    /// // Use migrations that were in a local folder during build: ./tests/migrations-1
    /// let v = Validator::from_migrator(sqlx::migrate!("./tests/migrations-1"));
    ///
    /// // Create a connection pool
    /// let pool = sqlx::sqlite::SqlitePoolOptions::new().connect("sqlite::memory:").await?;
    /// let mut conn = pool.acquire().await?;
    ///
    /// // Validate the migrations
    /// v.validate(&mut *conn).await?;
    /// # Ok(())
    /// # })
    /// # }
    /// ```
    async fn validate<'c, C>(&self, conn: &mut C) -> Result<(), ValidateError>
    where
        C: Migrate;
}

/// Validate previously applied migrations against the migration source.
/// Depending on the migration source this can be used to check if all migrations
/// for the current version of the application have been applied.
/// Use [`Validator::from_migrator`] to use the migrations available during compilation.
///
/// # Examples
///
/// ```rust,no_run
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// # sqlx_rt::block_on(async move {
/// # use sqlx_migrate_validate::Validator;
/// // Use migrations that were in a local folder during build: ./tests/migrations-1
/// let v = Validator::from_migrator(sqlx::migrate!("./tests/migrations-1"));
///
/// // Create a connection pool
/// let pool = sqlx::sqlite::SqlitePoolOptions::new().connect("sqlite::memory:").await?;
/// let mut conn = pool.acquire().await?;
///
/// // Validate the migrations
/// v.validate(&mut *conn).await?;
/// # Ok(())
/// # })
/// # }
/// ```
#[derive(Debug)]
pub struct Validator {
    pub migrations: Cow<'static, [Migration]>,
    pub ignore_missing: bool,
    pub locking: bool,
}

impl Validator {
    /// Creates a new instance with the given source. Please note that the source
    /// is resolved at runtime and not at compile time.
    /// You can use [`Validator::from<sqlx::Migrator>`] and the [`sqlx::migrate!`] macro
    /// to embed the migrations into the binary during compile time.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// # sqlx_rt::block_on(async move {
    /// # use sqlx_migrate_validate::Validator;
    /// use std::path::Path;
    ///
    /// // Read migrations from a local folder: ./tests/migrations-1
    /// let v = Validator::new(Path::new("./tests/migrations-1")).await?;
    ///
    /// // Create a connection pool
    /// let pool = sqlx::sqlite::SqlitePoolOptions::new().connect("sqlite::memory:").await?;
    /// let mut conn = pool.acquire().await?;
    ///
    /// // Validate the migrations
    /// v.validate(&mut *conn).await?;
    /// # Ok(())
    /// # })
    /// # }
    /// ```
    ///
    /// See [MigrationSource] for details on structure of the `./tests/migrations-1` directory.
    pub async fn new<'s, S>(source: S) -> Result<Self, MigrateError>
    where
        S: MigrationSource<'s>,
    {
        Ok(Self {
            migrations: Cow::Owned(source.resolve().await.map_err(MigrateError::Source)?),
            ignore_missing: false,
            locking: true,
        })
    }

    /// Creates a new instance with the migrations from the given migrator.
    /// You can combine this with the [`sqlx::migrate!`] macro
    /// to embed the migrations into the binary during compile time.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// # sqlx_rt::block_on(async move {
    /// # use sqlx_migrate_validate::Validator;
    /// // Use migrations that were in a local folder during build: ./tests/migrations-1
    /// let v = Validator::from_migrator(sqlx::migrate!("./tests/migrations-1"));
    ///
    /// // Create a connection pool
    /// let pool = sqlx::sqlite::SqlitePoolOptions::new().connect("sqlite::memory:").await?;
    /// let mut conn = pool.acquire().await?;
    ///
    /// // Validate the migrations
    /// v.validate(&mut *conn).await?;
    /// # Ok(())
    /// # })
    /// # }
    /// ```
    pub fn from_migrator(migrator: Migrator) -> Self {
        Self {
            migrations: migrator.migrations.clone(),
            ignore_missing: migrator.ignore_missing,
            locking: migrator.locking,
        }
    }

    pub async fn validate<'c, C>(&self, conn: &mut C) -> Result<(), ValidateError>
    where
        C: Migrate,
    {
        // lock the migrator to prevent other migrators from running
        if self.locking {
            conn.lock().await?;
        }

        let version = conn.dirty_version().await?;
        if let Some(version) = version {
            return Err(ValidateError::MigrateError(MigrateError::Dirty(version)));
        }

        let applied_migrations = conn.list_applied_migrations().await?;
        validate_applied_migrations(&applied_migrations, self)?;

        let applied_migrations: HashMap<_, _> = applied_migrations
            .into_iter()
            .map(|m| (m.version, m))
            .collect();

        for migration in self.migrations.iter() {
            if migration.migration_type.is_down_migration() {
                continue;
            }

            match applied_migrations.get(&migration.version) {
                Some(applied_migration) => {
                    if migration.checksum != applied_migration.checksum {
                        return Err(ValidateError::MigrateError(MigrateError::VersionMismatch(
                            migration.version,
                        )));
                    }
                }
                None => {
                    return Err(ValidateError::VersionNotApplied(migration.version));
                    // conn.apply(migration).await?;
                }
            }
        }

        // unlock the migrator to allow other migrators to run
        // but do nothing as we already migrated
        if self.locking {
            conn.unlock().await?;
        }

        Ok(())
    }
}

impl From<&Migrator> for Validator {
    fn from(migrator: &Migrator) -> Self {
        Self {
            migrations: migrator.migrations.clone(),
            ignore_missing: migrator.ignore_missing,
            locking: migrator.locking,
        }
    }
}

impl From<Migrator> for Validator {
    fn from(migrator: Migrator) -> Self {
        Self::from(&migrator)
    }
}

#[async_trait(?Send)]
impl Validate for Migrator {
    async fn validate<'c, C>(&self, conn: &mut C) -> Result<(), ValidateError>
    where
        C: Migrate,
    {
        Validator::from(self).validate(conn).await
    }
}

fn validate_applied_migrations(
    applied_migrations: &[AppliedMigration],
    migrator: &Validator,
) -> Result<(), MigrateError> {
    if migrator.ignore_missing {
        return Ok(());
    }

    let migrations: HashSet<_> = migrator.migrations.iter().map(|m| m.version).collect();

    for applied_migration in applied_migrations {
        if !migrations.contains(&applied_migration.version) {
            return Err(MigrateError::VersionMissing(applied_migration.version));
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use sqlx::migrate::MigrationType;

    use super::*;

    #[test]
    fn validate_applied_migrations_returns_ok_when_nothing_was_applied() {
        let applied_migrations = vec![];
        let mut validator = Validator {
            migrations: Cow::Owned(vec![]),
            ignore_missing: false,
            locking: true,
        };

        assert!(validate_applied_migrations(&applied_migrations, &validator).is_ok());

        validator.ignore_missing = true;
        assert!(validate_applied_migrations(&applied_migrations, &validator).is_ok());
    }

    #[test]
    fn validate_applied_migrations_returns_err_when_applied_migration_not_in_source() {
        let applied_migrations = vec![AppliedMigration {
            version: 1,

            // only the version is relevant for this method
            checksum: Cow::Owned(vec![]),
        }];
        let validator = Validator {
            migrations: Cow::Owned(vec![]),
            ignore_missing: false,
            locking: true,
        };

        match validate_applied_migrations(&applied_migrations, &validator) {
            Err(MigrateError::VersionMissing(i)) => assert_eq!(i, 1),
            _ => panic!("Unexpected error"),
        }
    }

    #[test]
    fn validate_applied_migrations_returns_ok_when_applied_migration_in_source() {
        let applied_migrations = vec![AppliedMigration {
            version: 1,

            // only the version is relevant for this method
            checksum: Cow::Owned(vec![]),
        }];
        let validator = Validator {
            migrations: Cow::Owned(vec![Migration {
                version: 1,

                // only the version is relevant for this method
                migration_type: MigrationType::ReversibleUp,
                checksum: Cow::Owned(vec![]),
                sql: Cow::Owned("".to_string()),
                description: Cow::Owned("".to_string()),
            }]),
            ignore_missing: false,
            locking: true,
        };

        match validate_applied_migrations(&applied_migrations, &validator) {
            Ok(_) => {}
            _ => panic!("Unexpected error"),
        }
    }
}
