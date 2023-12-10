//! Replace `~` with user home directory across all platforms
//!
//! Unit tests come from https://github.com/sathishsoundharajan/untildify (MIT LICENSE)

pub fn untildify(input_path: &str) -> String {
    if input_path.is_empty() {
        return String::from(input_path);
    }
    if let Some(home) = home::home_dir() {
        if input_path == r"~" {
            return home.into_os_string().into_string().unwrap();
        }
        if input_path.starts_with(r"~/") || input_path.starts_with(r"~\") {
            if let Ok(path) = home.join(&input_path[2..]).into_os_string().into_string() {
                return path;
            }
        }
    }
    String::from(input_path)
}

#[cfg(any(unix, target_os = "redox"))]
#[cfg(test)]
mod tests {
    use std::{env, path::Path};

    use super::*;

    #[ignore]
    #[test]
    fn test_returns_untildfyed_string() {
        env::remove_var("HOME");

        let home = Path::new("/User/Untildify");
        env::set_var("HOME", home.as_os_str());

        assert_eq!(untildify("~/Desktop"), "/User/Untildify/Desktop");
        assert_eq!(untildify("~/a/b/c/d/e"), "/User/Untildify/a/b/c/d/e");
        assert_eq!(untildify("~/"), "/User/Untildify/");
    }

    #[ignore]
    #[test]
    fn test_returns_empty_string() {
        env::remove_var("HOME");

        let home = Path::new("/User/Untildify");
        env::set_var("HOME", home.as_os_str());

        assert_eq!(untildify("Desktop"), "Desktop");
        assert_eq!(untildify(""), "");
        assert_eq!(untildify("/"), "/");
        // assert_eq!(untildify("~/Desktop/~/Code"), "/User/Untildify/Desktop/");
    }

    #[ignore]
    #[test]
    fn test_with_dot_folders() {
        env::remove_var("HOME");

        let home = Path::new("/User/Untildify");
        env::set_var("HOME", home.as_os_str());

        assert_eq!(untildify("~/.ssh/id_rsa"), "/User/Untildify/.ssh/id_rsa");
    }
}

#[cfg(target_os = "windows")]
#[cfg(test)]
mod tests {
    use std::env;

    use super::*;

    #[test]
    fn test_returns_untildfyed_string() {
        env::set_var("USERPROFILE", r"C:\Users\Admin");

        assert_eq!(untildify(r"~\Desktop"), r"C:\Users\Admin\Desktop");
        assert_eq!(untildify(r"~\a\b\c\d\e"), r"C:\Users\Admin\a\b\c\d\e");
        assert_eq!(untildify(r"~\"), r"C:\Users\Admin\");
    }
}
