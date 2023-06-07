# Self Hosting

## Configuration

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
