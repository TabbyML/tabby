# API Specs

## Repository api: `/repositories`

### Resolve

Get file or directory content from local repositories

**URL:** `/repositories/{kind}/{id}/resolve/{path}`

**Method:** `GET`
**Authorization:** Auth token used for GraphQL queries

**Request examples:**

- Get directory content

```shell
curl --request GET \
  --url http://localhost:8080/repositories/git/lNrAwW/resolve/ \
  -H "Authorization: Bearer {Token}"

curl --request GET \
  --url http://localhost:9090/repositories/github/E16n1q/resolve/ee/tabby-webserver/ \
  -H "Authorization: Bearer {Token}"
```

- Get file content

```shell
curl --request GET \
  --url http://localhost:8080/repositories/git/lNrAwW/resolve/package.json \
  -H "Authorization: Bearer {Token}"

curl --request GET \
  --url http://localhost:8080/repositories/git/lNrAwW/resolve/src/lib.rs \
  -H "Authorization: Bearer {Token}"
```

**Response examples:**

- All directory queries will return a list of strings, with each string representing an entry under that directory. The `Content-Type` for directory query is `application/vnd.directory+json`.

For `/repositories/git/lNrAwW/resolve/` (for a pre-populated repository), the response is:

```json
{
  "entries": [
    {
      "kind": "dir",
      "basename": "src"
    },
    {
      "kind": "file",
      "basename": "Cargo.toml"
    },
    {
      "kind": "file",
      "basename": ".gitignore"
    },
    {
      "kind": "file",
      "basename": "Cargo.lock"
    }
  ]
}
```

- The file query will return file content, the `Content-Type` will be guessed from the file extension.

For the request `/repositories/git/lNrAwW/resolve/package.json`, the content type is `application/json`, and the response is as follows (the content of package.json in the Tabby repository):

```json
{
  "private": true,
  "workspaces": [
    "clients/tabby-agent",
    "clients/vscode",
    "clients/vim",
    "clients/intellij",
    "clients/example-vscode-lsp"
  ],
  "engines": {
    "node": ">=18"
  }
}
```

For request `/repositories/git/lNrAwW/resolve/ee/tabby-webserver/src/lib.rs` (again in the Tabby repository), the content type is `text/x-rust`, and the response is:

```text
//! Defines behavior for the tabby webserver which allows users to interact with enterprise features.
//! Using the web interface (e.g chat playground) requires using this module with the `--webserver` flag on the command line.
mod axum;
mod hub;
mod jwt;
mod oauth;
mod path;
mod routes;
mod service;
mod webserver;

#[cfg(test)]
pub use service::*;

pub mod public {

    pub use super::{
        /* used by tabby workers (consumer of /hub api) */
        hub::{
            create_scheduler_client, create_worker_client, RegisterWorkerRequest, SchedulerClient,
            WorkerClient, WorkerKind,
        },
        webserver::Webserver,
    };
}
......omitted......
```

## OAuth api: `/oauth`

### List Providers

**URL:** `/oauth/providers`

**Method:** `GET`

**Request example:**

```shell
curl --request GET \
  --url http://localhost:8080/oauth/providers
```

**Response example:**

```json
["github"]
```

### SignIn

**URL:** `/oauth/signin`

**Method:** `GET`

**Request example:**

```shell
curl --request GET \
  --url http://localhost:8080/oauth/signin?provider=google
```

**Response example:**

Redirect to oauth provider for signin


### OAuth callback

**URL:** `/oauth/callback/{provider}`

**Method:** `GET`

**Request example:**

```shell
curl --request GET \
  --url http://localhost:8080/oauth/callback/github?code=1234567890
```

**Response example:**

The request will redirect to `/auth/signin` with refresh token & access token attached.

```
http://localhost:8080/auth/signin?refresh_token=321bc1bbb043456dae1a7abc0c447875&access_token=eyJ0eXAi......1NiJ9.eyJleHAi......bWluIjp0cnVlfQ.GvHSMUfc...S5BnwY
```

When an error occurs, the request will redirect to `/auth/signin` with error message & provider attached.
```
http://localhost:8080/auth/signin?error_msg=...&provider=github 
```
