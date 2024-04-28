use std::sync::Arc;

use anyhow::Result;
use async_stream::stream;
use async_trait::async_trait;
use futures::stream::BoxStream;
use minijinja::{context, Environment};
use tabby_common::api::chat::Message;
use tabby_inference::{
    chat::{self, ChatCompletionStream},
    TextGeneration, TextGenerationOptionsBuilder,
};

struct ChatPromptBuilder {
    env: Environment<'static>,
}

impl ChatPromptBuilder {
    pub fn new(prompt_template: String) -> Self {
        let mut env = Environment::new();
        env.add_function("raise_exception", |e: String| panic!("{}", e));
        env.add_template_owned("prompt", prompt_template)
            .expect("Failed to compile template");

        Self { env }
    }

    pub fn build(&self, messages: &[Message]) -> Result<String> {
        // System prompt is not supported for TextGenerationStream backed chat.
        let messages = messages
            .iter()
            .filter(|x| x.role != "system")
            .collect::<Vec<_>>();
        Ok(self.env.get_template("prompt")?.render(context!(
                messages => messages
        ))?)
    }
}

struct ChatCompletionImpl {
    engine: Arc<TextGeneration>,
    prompt_builder: ChatPromptBuilder,
}

#[async_trait]
impl ChatCompletionStream for ChatCompletionImpl {
    async fn chat_completion(
        &self,
        messages: &[Message],
        options: chat::ChatCompletionOptions,
    ) -> Result<BoxStream<String>> {
        let options = TextGenerationOptionsBuilder::default()
            .max_input_length(2048)
            .max_decoding_length(1920)
            .seed(options.seed)
            .sampling_temperature(options.sampling_temperature)
            .build()?;

        let prompt = self.prompt_builder.build(messages)?;

        let s = stream! {
            for await (streaming, content) in self.engine.generate_stream(&prompt, options).await {
                if streaming {
                    yield content
                }
            }
        };

        Ok(Box::pin(s))
    }
}

pub fn make_chat_completion(
    engine: Arc<TextGeneration>,
    prompt_template: String,
) -> impl ChatCompletionStream {
    ChatCompletionImpl {
        engine,
        prompt_builder: ChatPromptBuilder::new(prompt_template),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    static PROMPT_TEMPLATE : &str = "<s>{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if message['role'] == 'user' %}{{ '[INST] ' + message['content'] + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ message['content'] + '</s> ' }}{% else %}{{ raise_exception('Only user and assistant roles are supported!') }}{% endif %}{% endfor %}";

    #[test]
    fn test_it_works() {
        let builder = ChatPromptBuilder::new(PROMPT_TEMPLATE.to_owned());
        let messages = vec![
            Message {
                role: "user".to_owned(),
                content: "What is tail recursion?".to_owned(),
            },
            Message {
                role: "assistant".to_owned(),
                content: "It's a kind of optimization in compiler?".to_owned(),
            },
            Message {
                role: "user".to_owned(),
                content: "Could you share more details?".to_owned(),
            },
        ];
        assert_eq!(builder.build(&messages).unwrap(), "<s>[INST] What is tail recursion? [/INST]It's a kind of optimization in compiler?</s> [INST] Could you share more details? [/INST]")
    }
}
