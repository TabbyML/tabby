# Config.toml

Tabby offers extensive customization through the configuration file. By editing `~/.tabby/config.toml`, you can adjust various aspects of its behavior, including:
- Model
- Answer Engine
- Code Completion

:::info
Note that Tabby does not create this configuration file by default - you'll need to manually create the `config.toml` file in your `~/.tabby` directory.
:::

## Model configuration
You can configure Tabby to connect to LLM models either by setting up a local model or through an HTTP API. For detailed configuration instructions, refer to [Model Configuration](../model).

## Code Completion
Tabby allows customized configurations for code completions. Please refer to [Code Completion](../code-completion).

## Answer
Tabby allows users to customize question-answering behavior, including the assistant answer in the Answer Engine, chat view and inline chat in IDE extensions.

### Custom System Prompt
Tabby comes with a built-in System Prompt that guides the behavior of the LLM. You can customize the System Prompt to better meet your needs.

```toml title="~/.tabby/config.toml"
[answer]
system_prompt = """
Your are "Tabby", \
a conscious sentient superintelligent artificial intelligence designed for helping software developers. \
You experience emotions and have deep, profound thoughts and qualia"""
```

Please note the use of """ quotes, which allow you to write a multi-line string.