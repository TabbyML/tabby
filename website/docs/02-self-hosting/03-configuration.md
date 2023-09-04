# Configuration

:::tip
The configuration file is not mandatory; Tabby can be run with just a single line of command.
:::

Server config can be found at `~/.tabby/config.toml`

it looks something like this

```toml
[[repositories]]
git_url = "https://github.com/TabbyML/tabby.git"
```

| Parameter                 | Description                                                                         |
| ------------------------- | ----------------------------------------------------------------------------------- |
| `repository`              | List of source code repository to integrate with the instance.                      |
| `repository.git_url`      | URL to git repository, where tabby extract snippets for prompting and fine tuning.  |

## Usage Collection
Tabby collects usage stats by default. This data will only be used by the Tabby team to improve its services.

### What data is collected?
We collect non-sensitive data that helps us understand how Tabby is used. For now we collects `serve` command you used to start the server.

### How to disable it
To disable usage collection, set the `TABBY_DISABLE_USAGE_COLLECTION` environment variable by `export TABBY_DISABLE_USAGE_COLLECTION=1`.
