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

pub fn invitation_email(external_url: &str, code: &str) -> EmailContents {
    format_email(
        include_str!("templates/invitation.html"),
        &[("{external_url}", external_url), ("{code}", code)],
    )
}
