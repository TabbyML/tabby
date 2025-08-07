use std::collections::{HashMap, HashSet};

use async_trait::async_trait;
use juniper::{GraphQLInputObject, GraphQLObject};
use validator::{Validate, ValidateEmail, ValidationError};

use super::Result;

#[async_trait]
pub trait SettingService: Send + Sync {
    async fn read_security_setting(&self) -> Result<SecuritySetting>;
    async fn update_security_setting(&self, input: SecuritySettingInput) -> Result<()>;

    async fn read_network_setting(&self) -> Result<NetworkSetting>;
    async fn update_network_setting(&self, input: NetworkSettingInput) -> Result<()>;

    async fn read_branding_setting(&self) -> Result<BrandingSetting>;
    async fn update_branding_setting(&self, input: BrandingSettingInput) -> Result<()>;
}

#[derive(GraphQLObject, Debug, PartialEq)]
pub struct SecuritySetting {
    pub allowed_register_domain_list: Vec<String>,
    pub disable_client_side_telemetry: bool,
    pub disable_password_login: bool,
}

impl SecuritySetting {
    pub fn can_register_without_invitation(&self, email: &str) -> bool {
        self.allowed_register_domain_list
            .iter()
            .any(|domain| email.ends_with(&format!("@{domain}")))
    }
}

#[derive(GraphQLInputObject, Validate)]
pub struct SecuritySettingInput {
    #[validate(custom(function = "validate_unique_domains"))]
    pub allowed_register_domain_list: Vec<String>,
    pub disable_client_side_telemetry: bool,
    pub disable_password_login: bool,
}

#[derive(GraphQLObject, Debug, PartialEq)]
pub struct NetworkSetting {
    pub external_url: String,
}

#[derive(GraphQLInputObject, Validate)]
pub struct NetworkSettingInput {
    #[validate(url(code = "externalUrl", message = "URL is malformed"))]
    pub external_url: String,
}

#[derive(GraphQLObject, Debug, PartialEq)]
pub struct BrandingSetting {
    pub branding_logo: Option<String>,
    pub branding_icon: Option<String>,
}

#[derive(GraphQLInputObject, Validate)]
pub struct BrandingSettingInput {
    #[validate(custom(function = "validate_logo_image"))]
    pub branding_logo: Option<String>,
    #[validate(custom(function = "validate_icon_image"))]
    pub branding_icon: Option<String>,
}

fn first_duplicate(strings: &[impl std::hash::Hash + Eq]) -> Option<usize> {
    let mut set: HashSet<_> = Default::default();
    for (i, string) in strings.iter().enumerate() {
        if !set.insert(string) {
            return Some(i);
        }
    }
    None
}

fn validate_unique_domains(domains: &[String]) -> Result<(), ValidationError> {
    let duplicate_index = first_duplicate(domains);
    if let Some(duplicate_index) = duplicate_index {
        let err = ValidationError {
            code: format!("allowedRegisterDomainList.{duplicate_index}.value").into(),
            message: Some("Duplicate domain".into()),
            params: HashMap::default(),
        };
        return Err(err);
    }
    for (i, domain) in domains.iter().enumerate() {
        let email = format!("noreply@{domain}");
        if !email.validate_email() {
            let err = ValidationError {
                code: format!("allowedRegisterDomainList.{i}.value").into(),
                message: Some("Invalid domain".into()),
                params: HashMap::default(),
            };
            return Err(err);
        }
    }
    Ok(())
}

fn validate_logo_image(logo_image: &String) -> Result<(), ValidationError> {
    validate_image_impl(logo_image, "brandingLogo")
}

fn validate_icon_image(icon_image: &String) -> Result<(), ValidationError> {
    validate_image_impl(icon_image, "brandingIcon")
}

fn validate_image_impl(image: &String, code: &'static str) -> Result<(), ValidationError> {
    const MAX_IMAGE_SIZE_IN_BYTES: usize = 500 * 1024;
    // Base64 is about 33% larger than original.
    const MAX_BASE64_IMAGE_SIZE: usize = MAX_IMAGE_SIZE_IN_BYTES * 4 / 3 + 4;

    if image.is_empty() {
        return Ok(());
    }

    if image.len() > MAX_BASE64_IMAGE_SIZE {
        let mut err = ValidationError::new(code);
        err.message = Some("Max file size 500KB.".into());
        return Err(err);
    }

    let Some(mime_type) = image.split([',', ';']).next() else {
        let mut err = ValidationError::new(code);
        err.message = Some("Invalid image format".into());
        return Err(err);
    };

    if !matches!(
        mime_type,
        "data:image/png" | "data:image/jpeg" | "data:image/webp" | "data:image/svg+xml"
    ) {
        let mut err = ValidationError::new(code);
        err.message = Some("Accepted file types: .png, .jpg, .webp, .svg.".into());
        return Err(err);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::SecuritySetting;
    use crate::schema::setting::{first_duplicate, validate_unique_domains};

    #[test]
    fn test_validate_urls() {
        assert!(validate_unique_domains(&["example.com".to_owned()]).is_ok());

        assert!(validate_unique_domains(&["https://example.com".to_owned()]).is_err());

        assert!(validate_unique_domains(&["domain.withmultipleparts.com".to_owned()]).is_ok());

        assert!(
            validate_unique_domains(&["example.com".to_owned(), "example.com".to_owned()])
                .unwrap_err()
                .code
                .contains(".1.")
        );
    }

    #[test]
    fn test_duplicate_index() {
        assert_eq!(first_duplicate(&["a", "b", "c"]), None);
        assert_eq!(first_duplicate(&["a", "b", "b"]), Some(2));
        assert_eq!(first_duplicate(&["a", "b", "c", "c", "c"]), Some(3));
    }

    #[test]
    fn test_can_register_without_invitation() {
        let setting = SecuritySetting {
            allowed_register_domain_list: vec![],
            disable_client_side_telemetry: false,
            disable_password_login: false,
        };

        assert!(!setting.can_register_without_invitation("abc@abc.com"));

        let setting = SecuritySetting {
            allowed_register_domain_list: vec!["".into()],
            disable_client_side_telemetry: false,
            disable_password_login: false,
        };

        assert!(!setting.can_register_without_invitation("abc@abc.com"));

        let setting = SecuritySetting {
            allowed_register_domain_list: vec![".com".into()],
            disable_client_side_telemetry: false,
            disable_password_login: false,
        };

        assert!(!setting.can_register_without_invitation("abc@abc.com"));

        let setting = SecuritySetting {
            allowed_register_domain_list: vec!["abc.com".into()],
            disable_client_side_telemetry: false,
            disable_password_login: false,
        };

        assert!(setting.can_register_without_invitation("abc@abc.com"));
    }
}
