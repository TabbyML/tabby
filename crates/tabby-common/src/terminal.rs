pub enum HeaderFormat {
    BoldWhite,
    BoldBlue,
    BoldYellow,
    BoldRed,
    Blue,
}

impl HeaderFormat {
    fn prefix(&self) -> &str {
        match self {
            HeaderFormat::BoldWhite => "\x1b[1m",
            HeaderFormat::BoldBlue => "\x1b[34;1m",
            HeaderFormat::BoldYellow => "\x1b[93;1m",
            HeaderFormat::Blue => "\x1b[34m",
            HeaderFormat::BoldRed => "\x1b[1;31m",
        }
    }

    pub fn format(&self, header: &str) -> String {
        format!("{}{header}\x1b[0m", self.prefix())
    }
}

pub struct InfoMessage<'a> {
    header: &'a str,
    header_format: HeaderFormat,
    lines: &'a [&'a str],
}

impl<'a> InfoMessage<'a> {
    pub fn new(header: &'a str, header_format: HeaderFormat, lines: &'a [&'a str]) -> Self {
        Self {
            header,
            header_format,
            lines,
        }
    }

    pub fn print(self) {
        eprintln!("\n{}\n", self.to_string());
    }

    pub fn print_messages(messages: &[Self]) {
        let messages: Vec<String> = messages.iter().map(|m| m.to_string()).collect();
        eprintln!("\n{}\n", messages.join("\n"));
    }
}

impl ToString for InfoMessage<'_> {
    fn to_string(&self) -> String {
        let mut str = String::new();
        str.push_str(&format!("  {}\n\n", self.header_format.format(self.header)));
        for (i, line) in self.lines.iter().enumerate() {
            str.push_str("  ");
            str.push_str(line);
            if i != self.lines.len() + 1 {
                str.push('\n');
            }
        }
        str
    }
}
