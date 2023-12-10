use std::{fs, io, str};

use sha2::{Digest, Sha256};

use crate::error::ValidateError;

pub struct HashChecker;
impl HashChecker {
    pub fn check(filename: &str, expected_hash: &str) -> Result<(), ValidateError> {
        let mut result = Ok(());
        if filename != "stdout" && (!expected_hash.is_empty()) {
            let actual_hash = HashChecker::sha256sum(filename);
            if actual_hash != expected_hash {
                result = Err(ValidateError::Sha256Mismatch);
            }
            match result {
                Ok(()) => println!("✅ Checksum OK."),
                Err(ValidateError::Sha256Mismatch) => println!(
                    "❌ Checksum verification failed for {filename}:\n  expected: {expected_hash}\n  got:      {actual_hash}"),
            }
        }
        result
    }

    fn sha256sum(filename: &str) -> String {
        let mut hasher = Sha256::new();
        let mut file = fs::File::open(filename).unwrap();

        io::copy(&mut file, &mut hasher).unwrap();
        let computed_hash = hasher.finalize();
        drop(file);

        format!("{computed_hash:x}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    macro_rules! assert_err {
        ($expression:expr, $($pattern:tt)+) => {
            match $expression {
                $($pattern)+ => (),
                ref e => panic!("expected `{}` but got `{:?}`", stringify!($($pattern)+), e),
            }
        }
    }

    #[test]
    fn test_check_api_fails_when_checksum_mismatch() {
        let expected = "AAAA847124bfb9d9a9d44af6f00d8003006c44b9ef9ba458b5d4d3fc5f81bde5";

        assert_err!(
            HashChecker::check("LICENCE.md", expected),
            Err(ValidateError::Sha256Mismatch)
        );
    }

    #[test]
    fn test_sha256sum_api() {
        let expected = "21d7847124bfb9d9a9d44af6f00d8003006c44b9ef9ba458b5d4d3fc5f81bde5";

        let actual = HashChecker::sha256sum("LICENCE.md");

        assert_eq!(actual, expected);
    }

    #[test]
    fn test_check_api_works_when_typical() {
        let expected = "21d7847124bfb9d9a9d44af6f00d8003006c44b9ef9ba458b5d4d3fc5f81bde5";

        let is_match = HashChecker::check("LICENCE.md", expected).is_ok();

        assert!(is_match);
    }
}
