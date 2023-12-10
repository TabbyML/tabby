use url_parse::core::Parser;

use crate::netrc::netrc;

#[derive(Debug)]
pub struct ParsedAddress {
    pub server: String,
    pub username: String,
    pub password: String,
    pub path_segments: Vec<String>,
    pub file: String,
}

impl PartialEq for ParsedAddress {
    fn eq(&self, other: &Self) -> bool {
        let result = self.server == other.server
            && self.username == other.username
            && self.password == other.password
            && self.file == other.file;

        let mut paths_equal = true;
        for it in self.path_segments.iter().zip(self.path_segments.iter()) {
            let (left, right) = it;
            paths_equal = paths_equal && (left == right);
        }

        result && paths_equal
    }
}

impl ParsedAddress {
    pub fn parse_address(address: &str, silent: bool) -> ParsedAddress {
        let netrc = netrc(silent);
        let url = Parser::new(None).parse(address).unwrap();
        let server = format!(
            "{}:{}",
            url.host_str()
                .ok_or_else(|| panic!("failed to parse hostname from url: {url}"))
                .unwrap(),
            url.port_or_known_default()
                .ok_or_else(|| panic!("failed to parse port from url: {url}"))
                .unwrap(),
        );

        let url_username = url.username();
        let username = if url_username.is_none() {
            "anonymous".to_string()
        } else {
            url.username().unwrap()
        };

        let password = url.password().unwrap_or_else(|| "anonymous".to_string());
        if !silent && username != "anonymous" && password != "anonymous" {
            println!("🔑 Parsed credentials from URL.");
        }

        let (username, password) = ParsedAddress::mixin_netrc(&netrc, &server, username, password);

        let mut path_segments: Vec<String> = url
            .path_segments()
            .ok_or_else(|| panic!("failed to get url path segments: {url}"))
            .unwrap();

        let file = path_segments
            .pop()
            .ok_or_else(|| panic!("got empty path segments from url: {url}"))
            .unwrap();

        ParsedAddress {
            server,
            username,
            password,
            path_segments,
            file,
        }
    }

    fn mixin_netrc(
        netrc: &Option<netrc::Netrc>,
        server: &str,
        username: String,
        password: String,
    ) -> (String, String) {
        let mut user = username.clone();
        let mut pass = password.clone();
        if !netrc.is_none() && username == "anonymous" && password == "anonymous" {
            for host in netrc.as_ref().unwrap().hosts.iter().enumerate() {
                let (_i, (netrc_name, machine)) = host;

                let mut name = netrc_name.to_string();
                if let Some(port) = machine.port {
                    name = name + ":" + &port.to_string()[..];
                }
                if server == name {
                    user = machine.login.clone();
                    pass = machine.password.clone().unwrap();
                    break;
                }
            }
        }
        (user, pass)
    }
}

#[tokio::test]
async fn parseaddress_operator_equals_works_when_typical() {
    let left = ParsedAddress {
        server: "do.main".to_string(),
        username: "user".to_string(),
        password: "pass".to_string(),
        path_segments: vec!["my".to_string(), "path".to_string()],
        file: "pass".to_string(),
    };
    let right = ParsedAddress {
        server: "do.main".to_string(),
        username: "user".to_string(),
        password: "pass".to_string(),
        path_segments: vec!["my".to_string(), "path".to_string()],
        file: "pass".to_string(),
    };

    assert!(left == right);
}

#[tokio::test]
async fn parseaddress_operator_equals_fails_when_not_equal() {
    let left = ParsedAddress {
        server: "do.main".to_string(),
        username: "user".to_string(),
        password: "pass".to_string(),
        path_segments: vec!["my".to_string(), "path".to_string()],
        file: "pass".to_string(),
    };
    let right = ParsedAddress {
        server: "do".to_string(),
        username: "user".to_string(),
        password: "pass".to_string(),
        path_segments: vec!["my".to_string(), "path".to_string()],
        file: "pass".to_string(),
    };

    assert!(left != right);
}

#[tokio::test]
async fn parse_works() {
    let expected = ParsedAddress {
        server: "do.main:21".to_string(),
        username: "user".to_string(),
        password: "pass".to_string(),
        path_segments: vec!["index".to_string()],
        file: "file".to_string(),
    };

    let actual = ParsedAddress::parse_address("ftp://user:pass@do.main:21/index/file", true);

    assert_eq!(actual, expected);
}

#[tokio::test]
async fn mixin_works() {
    let expected_username = "test";
    let expected_password = "p@ssw0rd";
    let input = "machine example.com login test password p@ssw0rd";
    let input = std::io::BufReader::new(input.as_bytes());
    let netrc = netrc::Netrc::parse(input).unwrap();
    let username_decoded_from_url = "anonymous".to_string();
    let password_decoded_from_url = "anonymous".to_string();

    let (actual_username, actual_password) = ParsedAddress::mixin_netrc(
        &Some(netrc),
        "example.com",
        username_decoded_from_url,
        password_decoded_from_url,
    );

    assert_eq!(actual_username, expected_username);
    assert_eq!(actual_password, expected_password);
}

#[tokio::test]
async fn mixin_works_with_port() {
    let expected_username = "test";
    let expected_password = "p@ssw0rd";
    let input = "machine example.com login test password p@ssw0rd port 443";
    let input = std::io::BufReader::new(input.as_bytes());
    let netrc = netrc::Netrc::parse(input).unwrap();
    let username_decoded_from_url = "anonymous".to_string();
    let password_decoded_from_url = "anonymous".to_string();

    let (actual_username, actual_password) = ParsedAddress::mixin_netrc(
        &Some(netrc),
        "example.com:443",
        username_decoded_from_url,
        password_decoded_from_url,
    );

    assert_eq!(actual_username, expected_username);
    assert_eq!(actual_password, expected_password);
}

#[tokio::test]
async fn parse_works_with_netrc_mixin() {
    let expected = ParsedAddress {
        server: "do.main:21".to_string(),
        username: "test".to_string(),
        password: "p@ssw0rd".to_string(),
        path_segments: vec!["index".to_string()],
        file: "file".to_string(),
    };
    let data = "machine do.main login test password p@ssw0rd port 21";

    std::fs::write(".netrc.test", data).expect("Unable to write file");
    let actual = ParsedAddress::parse_address("ftp://do.main/index/file", true);

    assert_eq!(actual, expected);
    std::fs::remove_file(".netrc.test").unwrap();
}

#[tokio::test]
async fn parse_works_when_ssh_user() {
    let expected = ParsedAddress {
        server: "localhost:2223".to_string(),
        username: "user".to_string(),
        password: "anonymous".to_string(),
        path_segments: vec!["".to_string()],
        file: "file".to_string(),
    };

    let actual = ParsedAddress::parse_address("ssh://user@localhost:2223/file", true);

    assert_eq!(actual, expected);
}

#[tokio::test]
async fn parse_works_when_not_silent() {
    let expected = ParsedAddress {
        server: "localhost:2223".to_string(),
        username: "user".to_string(),
        password: "pass".to_string(),
        path_segments: vec!["".to_string()],
        file: "file".to_string(),
    };

    let actual = ParsedAddress::parse_address("ssh://user:pass@localhost:2223/file", false);

    assert_eq!(actual, expected);
}
