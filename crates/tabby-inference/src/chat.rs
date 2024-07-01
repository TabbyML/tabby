use async_openai::config::OpenAIConfig;

pub trait ChatCompletionStream: Sync + Send {
    fn get(&self) -> async_openai::Chat<'_, OpenAIConfig>;
}

impl ChatCompletionStream for async_openai::Client<OpenAIConfig> {
    fn get(&self) -> async_openai::Chat<'_, OpenAIConfig> {
        self.chat()
    }
}
