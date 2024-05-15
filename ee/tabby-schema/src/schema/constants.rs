use lazy_static::lazy_static;
use regex::Regex;

lazy_static! {
    pub static ref REPOSITORY_NAME_REGEX: Regex = Regex::new("^[a-zA-Z][\\w.-]+$").unwrap();
    pub static ref USERNAME_REGEX: Regex = Regex::new("^[A-Za-z][A-Za-z ]*[A-Za-z]$").unwrap();
}
