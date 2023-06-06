# Configuration

Tabby maintains its runtime data in `~/.tabby`, with `~/.tabby/config.toml` serving as its configuration file.

You can modify the path to the data directory by setting the `TABBY_ROOT` variable. For example:

```
export TABBY_ROOT=/data/tabby
```

## Example `~/.tabby/config.toml`

```toml
[[repositories]]
git_url = "https://github.com/TabbyML/tabby.git"
```

### `[[repositories]]`

Git repositories integrated into Tabby for prompting/fine-tuning.
