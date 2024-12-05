use lazy_static::lazy_static;
use regex::Regex;

lazy_static! {
    pub static ref REPOSITORY_NAME_REGEX: Regex = Regex::new("^[a-zA-Z][\\w.-]+$").unwrap();
    pub static ref USERNAME_REGEX: Regex =
        Regex::new(r"^[^0-9±!@£$%^&*_+§¡€#¢¶•ªº«\\/<>?:;|=.,]{2,20}$").unwrap();
    pub static ref WEB_DOCUMENT_NAME_REGEX: Regex =
        Regex::new(r"^[A-Za-z][A-Za-z0-9#]*(?:[\s.-][A-Za-z0-9]+)*$").unwrap();
    pub static ref USER_GROUP_NAME_REGEX: Regex = Regex::new(r"^[a-z][a-z0-9_-]*$").unwrap();
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

    #[test]
    fn test_web_document_name_regex() {
        let test_cases = vec![
            ("John", true),    // English name
            ("Müller", false), // German name
            ("abc123", true),
            ("Abc 123", true),
            (" abc 123", false),
            ("abc123*", false),
            ("abc123_", false),
            ("abc  123", false), // two space
            ("ABC-abc", true),
        ];

        for (name, expected) in test_cases {
            let result = WEB_DOCUMENT_NAME_REGEX.is_match(name);
            assert_eq!(result, expected, "Failed for name: {}", name);
        }
    }

    #[test]
    fn test_user_group_name_regex() {
        let test_cases = vec![
            ("group", true),       // Valid name
            ("group123", true),    // Valid name with numbers
            ("group-name", true),  // Valid name with hyphen
            ("group_name", true),  // Valid name with underscore
            ("Group", false),      // Invalid: starts with uppercase letter
            ("1group", false),     // Invalid: starts with number
            ("group name", false), // Invalid: contains space
            ("group*name", false), // Invalid: contains special character
        ];

        for (name, expected) in test_cases {
            let result = USER_GROUP_NAME_REGEX.is_match(name);
            assert_eq!(result, expected, "Failed for name: {}", name);
        }
    }
}
