# API Specs

## Repository api: `/repositories`

### Resolve

Get file or directory content from local repositories

**URL:** `/repositories/{name}/resolve/{path}`

**Method:** `GET`

**Request examples:**

- Get directory content

```shell
curl --request GET \
  --url http://localhost:8080/repositories/https_github.com_TabbyML_tabby.git/resolve/

curl --request GET \
  --url http://localhost:9090/repositories/https_github.com_TabbyML_tabby.git/resolve/ee/tabby-webserver/
```

- Get file content

```shell
curl --request GET \
  --url http://localhost:8080/repositories/https_github.com_TabbyML_tabby.git/resolve/package.json

curl --request GET \
  --url http://localhost:9090/repositories/https_github.com_TabbyML_tabby.git/resolve/ee/tabby-webserver/src/api.rs
```

**Response examples:**

- All directory query will return a list of string, with each string represents an entry under that directory. The `Content-Type` for directory query is `application/vnd.directory+json`.

For `/repositories/https_github.com_TabbyML_tabby.git/resolve/ee/tabby-webserver/`, the response is:

```json
{
  "entries": [
    "ee/tabby-webserver/src",
    "ee/tabby-webserver/ui",
    "ee/tabby-webserver/examples",
    "ee/tabby-webserver/Cargo.toml",
    "ee/tabby-webserver/graphql"
  ]
}
```

- The file query will return file content, the `Content-Type` will be guessed from the file extension.

For request `/repositories/https_github.com_TabbyML_tabby.git/resolve/package.json`, the content type is `application/json`, and the response is:

```json
{
  "private": true,
  "workspaces": [
    "clients/tabby-agent",
    "clients/vscode",
    "clients/vim",
    "clients/intellij"
  ],
  "engines": {
    "node": ">=18"
  }
}
```

For request `/repositories/https_github.com_TabbyML_tabby.git/resolve/ee/tabby-webserver/src/api.rs`, the content type is `text/x-rust`, and the response is:

```text
use async_trait::async_trait;
use juniper::{GraphQLEnum, GraphQLObject};
use serde::{Deserialize, Serialize};
use tabby_common::api::{
    code::{CodeSearch, CodeSearchError, SearchResponse},
    event::RawEventLogger,
};
use thiserror::Error;
use tokio_tungstenite::connect_async;

use crate::websocket::WebSocketTransport;

#[derive(GraphQLEnum, Serialize, Deserialize, Clone, Debug)]
pub enum WorkerKind {
    Completion,
    Chat,
}

......omit......
```

### Meta

Get dataset entry for each indexed file in the repository

**URL:** `/repositories/{name}/meta/{path}`

**Method:** `GET`

**Request example:**

```shell
curl --request GET \
  --url http://localhost:9090/repositories/https_github.com_TabbyML_tabby.git/meta/ee/tabby-webserver/src/lib.rs
```

**Response example:**

The `Content-Type` for successful response is always `application/json`.

```json
{
  "git_url": "https://github.com/TabbyML/tabby.git",
  "filepath": "ee/tabby-webserver/src/lib.rs",
  "language": "rust",
  "max_line_length": 88,
  "avg_line_length": 26.340782,
  "alphanum_fraction": 0.56416017,
  "tags": [
    {
      "range": {
        "start": 0,
        "end": 12
      },
      "name_range": {
        "start": 8,
        "end": 11
      },
      "line_range": {
        "start": 0,
        "end": 12
      },
      "is_definition": true,
      "syntax_type_name": "module"
    },
    ......omit......
  ]
}
```
