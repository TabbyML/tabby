extern crate custom_error;
use custom_error::custom_error;

custom_error! {
    pub ValidateError
    Sha256Mismatch = "Invalid sha256.",
}

custom_error! {
    pub DownloadError
    Validate {source: ValidateError} = "{source}",
    HttpError {name: String, code: String} = "Failed to download {name}: Server returned {code}",
}

custom_error! {
    pub HTTPHeaderError
    NotPresent = "Cannot find requested header.",
}

impl From<ValidateError> for std::io::Error {
    fn from(cause: ValidateError) -> std::io::Error {
        std::io::Error::other(cause.to_string())
    }
}

#[test]
fn test_from_validate_error_to_std_io_error_works_when_typical() {
    let _ = match Err(ValidateError::Sha256Mismatch) {
        Ok(v) => v,
        Err(e) => {
            let error: std::io::Error = e.into();
            error
        }
    };
}
