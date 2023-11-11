use minijinja::{context, Environment};

use super::Message;

pub struct ChatPromptBuilder {
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

    pub fn build(&self, messages: &[Message]) -> String {
        self.env
            .get_template("prompt")
            .unwrap()
            .render(context!(
                    messages => messages
            ))
            .expect("Failed to evaluate")
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
        assert_eq!(builder.build(&messages), "<s>[INST] What is tail recursion? [/INST]It's a kind of optimization in compiler?</s> [INST] Could you share more details? [/INST]")
    }

    #[test]
    #[should_panic]
    fn test_it_panic() {
        let builder = ChatPromptBuilder::new(PROMPT_TEMPLATE.to_owned());
        let messages = vec![Message {
            role: "system".to_owned(),
            content: "system".to_owned(),
        }];
        builder.build(&messages);
    }
}
