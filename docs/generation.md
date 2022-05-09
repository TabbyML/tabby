# Text generation

CTranslate2 exposes high-level classes to run generative language models such as [GPT-2](https://github.com/openai/gpt-2). The main entrypoint is the [`Generator`](python/ctranslate2.Generator.rst) class which can run conditional or unconditional generation.

```{tip}
See the [Transformers](guides/transformers.md) guide which includes an example using GPT-2.
```

## Special tokens

Special tokens such as the decoder start token `<s>` should be explicitly included in the input if required by the model. No special tokens are added by the generator methods.

```{note}
This is different from the translator methods which usually include these special tokens implicitly.
```
