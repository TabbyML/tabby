use std::{io::BufReader, path::Path};

use netrc::Netrc;

use crate::untildify::untildify;

fn get_possible_netrc_path(silent: bool) -> String {
    let candidates = vec![
        ".netrc",
        ".netrc.test",
        ".netrc.test_https",
        ".netrc.test_ftp",
        ".netrc.test_unit",
        "~/.netrc",
    ];
    for candidate in candidates {
        let candidate = untildify(candidate);
        if Path::new(&candidate).exists() {
            if !silent {
                println!("🔑 Parsed .netrc from: {candidate}");
            }
            return candidate;
        }
    }
    "".to_string()
}

pub fn netrc(silent: bool) -> Option<netrc::Netrc> {
    let mut result = None;
    let path = get_possible_netrc_path(silent);
    if !path.is_empty() {
        let file = std::fs::File::open(path).unwrap();
        let parsed = Netrc::parse(BufReader::new(file));
        result = Some(parsed.unwrap());
    }
    result
}

#[test]
fn test_netrc_with_file_works_when_typical() {
    use std::io::Write;
    let mut file = std::fs::File::create(".netrc.test_unit").unwrap();
    file.write_all(b"machine mydomain.com login myuser password mypass port 1234")
        .unwrap();

    assert!(netrc(true).is_some());
    assert!(netrc(false).is_some());

    std::fs::remove_file(".netrc.test_unit").unwrap();
}

#[ignore]
#[test]
fn test_netrc_with_file_works_when_typical_and_not_silent() {
    use std::io::Write;
    let mut file = std::fs::File::create(".netrc.test_unit").unwrap();
    file.write_all(b"machine mydomain.com login myuser password mypass port 1234")
        .unwrap();

    let netrc = netrc(false);

    assert!(netrc.is_some());
    std::fs::remove_file(".netrc.test_unit").unwrap();
}
