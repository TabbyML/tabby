use octocrab;

pub fn octocrab_error_message(err: octocrab::Error) -> String {
    match err {
        octocrab::Error::GitHub { source, .. } => {
            format!("GitHub error: {}, {}", source.status_code, source.message)
        }

        // no need to print the Json error backtrace
        octocrab::Error::Json { source, .. } => {
            format!("Json error: {source}")
        }

        // the other errors have impl Display or Debug
        _ => err.to_string(),
    }
}
