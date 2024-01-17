pub enum HeaderFormat {
    BoldWhite,
    BoldBlue,
    BoldYellow,
    Blue,
}

impl HeaderFormat {
    fn prefix(&self) -> &str {
        match self {
            HeaderFormat::BoldWhite => "\x1b[1m",
            HeaderFormat::BoldBlue => "\x1b[34;1m",
            HeaderFormat::BoldYellow => "\x1b[93;1m",
            HeaderFormat::Blue => "\x1b[34m",
        }
    }

    pub fn format(&self, header: &str) -> String {
        format!("{}{header}\x1b[0m", self.prefix())
    }
}

pub fn show_info(header: &str, style: HeaderFormat, content: &[&str]) {
    eprintln!("  {}", style.format(header));
    for line in content {
        eprintln!("  {line}");
    }
    eprintln!();
}
