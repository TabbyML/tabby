# Amazon Bedrock

Amazon Bedrock is a fully managed service on AWS that provides access to foundation models from various AI companies through a single API. With [Amazon Bedrock Access Gateway](https://github.com/aws-samples/bedrock-access-gateway), you can access Anthropic's Claude models through an OpenAI-compatible interface, enabling seamless integration with tools and applications designed for OpenAI's API structure.

Follow the Amazon Bedrock Access Gateway setup guide to deploy your own OpenAI-compatible API endpoint for Claude models.

## Chat model

Amazon Bedrock Access Gateway provides an OpenAI-compatible chat API interface for Claude models. Here we use the `us.anthropic.claude-3-5-sonnet-20241022-v2:0` model as an example.

```toml title="~/.tabby/config.toml"
[[endpoints]]
name = "bedrock_chat"
api_route = "http://Bedrock-Proxy-xxxxx.{Region}.elb.amazonaws.com/api"
headers = {
  Authorization = "Bearer your-api-key"
}
metadata = {
  pochi = {
    use_case = "chat",
    provider = "openai",
    models = [
      { name = "us.anthropic.claude-3-5-sonnet-20241022-v2:0", context_window = 200000 }
    ]
  }
}
```

## Completion model

Amazon Bedrock does not provide completion models.
