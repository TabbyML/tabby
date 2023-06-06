# Server

## Docker

We recommend adding the following aliases to your `.bashrc` or `.zshrc` file:

```shell
# Save aliases to bashrc / zshrc
alias tabby="docker run -u $(id -u) -p 8080:8080 -v $HOME/.tabby:/data tabbyml/tabby"

# Alias for GPU (requires NVIDIA Container Toolkit)
alias tabby-gpu="docker run --gpus all -u $(id -u) -p 8080:8080 -v $HOME/.tabby:/data tabbyml/tabby"
```

After adding these aliases, you can use the `tabby` command as usual. Here are some examples of its usage:

```shell
# Usage
tabby --help

# Serve the model
tabby serve --model TabbyML/J-350M

# Serve with cuda
tabby-gpu --model TabbyML/J-350M --device cuda
```

## Configuration

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
