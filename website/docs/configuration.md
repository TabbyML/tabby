# ⚙️ Configuration

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
As of the date 10/07/2023, the following information has been collected:

```rust
struct HealthState {
    model: String,
    chat_model: Option<String>,
    device: String,
    arch: String,
    cpu_info: String,
    cpu_count: usize,
    cuda_devices: Vec<String>,
    version: Version,
}
```

For an up-to-date list of the fields we have collected, please refer to [health.rs](https://github.com/TabbyML/tabby/blob/main/crates/tabby/src/serve/health.rs#L11).

### How to disable it
To disable usage collection, set the `TABBY_DISABLE_USAGE_COLLECTION` environment variable by `export TABBY_DISABLE_USAGE_COLLECTION=1`.
