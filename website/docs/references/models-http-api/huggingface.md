# Hugging Face Inference Providers

[Hugging Face Inference Providers](https://huggingface.co/docs/inference-providers) offers access to frontier open models from multiple providers through a unified API. 

You'll need a [Hugging Face account](https://huggingface.co/join) and an [access token](https://huggingface.co/settings/tokens/new?ownUserPermissions=inference.serverless.write&tokenType=fineGrained).

## Chat model

Hugging Face Inference Providers provides an OpenAI-compatible chat API interface. Here we use the `MiniMaxAI/MiniMax-M2` model as an example.

```toml title="~/.tabby/config.toml"
[model.chat.http]
kind = "openai/chat"
model_name = "MiniMaxAI/MiniMax-M2" # specify the model you want to use
api_endpoint = "https://router.huggingface.co/v1"
api_key = "your-hf-token"
```

### Available models

You can find a complete list of models supported by at least one provider [on the Hub](https://huggingface.co/models?inference_provider=all). You can also access these programmatically, see this [guide](https://huggingface.co/docs/inference-providers/hub-api) for more details.

## Completion model

Hugging Face Inference Providers does not offer completion models (FIM) through their OpenAI-compatible API. For code completion, use a local model with Tabby.

## Embeddings model

While Hugging Face Inference Providers supports embeddings models, Tabby does not currently support the embeddings API interface for Hugging Face Inference Providers.
