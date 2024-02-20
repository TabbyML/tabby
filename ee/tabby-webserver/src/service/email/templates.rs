pub struct EmailContents {
    pub subject: String,
    pub body: String,
}

fn format_email(template: &'static str, replacements: &[(&str, &str)]) -> EmailContents {
    let (subject, body) = template
        .split_once("---")
        .expect("Email template must have subject and body separated by ---");
    let (mut subject, mut body) = (subject.to_string(), body.to_string());
    for (name, replacement) in replacements {
        body = body.replace(name, replacement);
        subject = subject.replace(name, replacement);
    }
    EmailContents { subject, body }
}

macro_rules! template_email {
    ($lit:ident: $($arg:ident),*) => {
        {
            let contents = include_str!(concat!(
                "../../../email_templates/",
                stringify!($lit),
                ".html"
            ));
            format_email(contents, &[
                $(
                    (&format!("{{{{{}}}}}", stringify!($arg).to_uppercase()), $arg)
                ),*
            ])
        }
    };
}

pub fn invitation(external_url: &str, code: &str) -> EmailContents {
    template_email!(invitation: external_url, code)
}

pub fn test() -> EmailContents {
    template_email!(test: )
}

pub fn password_reset(external_url: &str, email: &str, code: &str) -> EmailContents {
    template_email!(password_reset: external_url, email, code)
}
