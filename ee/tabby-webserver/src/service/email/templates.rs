pub struct EmailContents {
    pub subject: String,
    pub body: String,
}

fn format_email(template: &'static str, replacements: &[(&str, &str)]) -> EmailContents {
    let mut lines = template.lines();
    let mut subject = lines
        .next()
        .expect("Email must have subject line")
        .to_string();
    let body: Vec<&str> = lines.collect();
    let mut body = body.join("\n");
    for (name, replacement) in replacements {
        body = body.replace(name, replacement);
        subject = subject.replace(name, replacement);
    }
    EmailContents { subject, body }
}

macro_rules! template_email {
    ($lit:ident: $($arg:ident),*) => {
        pub fn $lit($($arg: &str),*) -> EmailContents {
            let contents = include_str!(concat!(
                "../../../email_templates/",
                stringify!($lit),
                ".html"
            ));
            format_email(contents, &[
                $(
                    (concat!("{", stringify!($arg), "}"), $arg)
                ),*
            ])
        }
    };
}

template_email!(invitation: external_url, code);
template_email!(test: );
