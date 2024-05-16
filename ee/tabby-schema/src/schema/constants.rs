use lazy_static::lazy_static;
use regex::Regex;

lazy_static! {
    pub static ref REPOSITORY_NAME_REGEX: Regex = Regex::new("^[a-zA-Z][\\w.-]+$").unwrap();
    pub static ref USERNAME_REGEX: Regex =
        Regex::new(r"^[^0-9±!@£$%^&*_+§¡€#¢¶•ªº«\\/<>?:;|=.,]{2,20}$").unwrap();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_username_regex() {
        let test_cases = vec![
            ("John", true),      // English name
            ("Müller", true),    // German name
            ("Jørgensen", true), // Danish name
            ("李雷", true),      // Chinese name
            ("あきは", true),    // Japanese name
            ("김민수", true),    // Korean name
            ("Алексей", true),   // Russian name
            ("José", true),      // Spanish names
            ("علی", true),       // Iranian names
            // Edge cases
            ("", false),                         // Empty string
            ("JohnDoeIsAReallyLongName", false), // More than 20 characters
            ("John!", false),                    // Invalid character '!'
            ("José@", false),                    // Invalid character '@'
            ("12345", false),                    // Invalid character Numbers
            ("John_Doe", false),                 // Underscore character
            ("Anna-Marie", true),                // Hyphen character
            ("O'Connor", true),                  // Apostrophe
            ("李@伟", false),
        ];

        for (name, expected) in test_cases {
            let result = USERNAME_REGEX.is_match(name);
            assert_eq!(result, expected, "Failed for name: {}", name);
        }
    }
}
