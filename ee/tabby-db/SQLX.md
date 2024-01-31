# Using SQLX with Tabby

Tabby uses SQLX for its database, which comes with several advantages:

- All queries created using the `query!()`, `query_as!()` and `query_scalar!()` macros are syntax- and schema-checked at compiletime
- All migrations are managed by sqlx
- A local copy of our db schema is maintained in `ee/tabby-db/schema.sqlite`
- Simplified querying syntax compared to rusqlite, which was previously used

To work with SQLX, it is recommended that you install it as a cargo subcommand:

```
cargo install sqlx-cli
```

This will allow you to use sqlx's commands like `cargo sqlx`.

To create the database fresh from migrations:

```
rm ee/tabby-db/schema.sqlite
cargo sqlx db setup --source ee/tabby-db/migrations
```

To create a new migration:

```
cargo sqlx migrate add --source ee/tabby-db/migrations -r -s <migration name>
```

This will create a new `up` and `down` file for the migration.

To run all migrations and ensure the local schema is up-to-date:

```
cargo sqlx migrate run --source ee/tabby-db/migrations
```
