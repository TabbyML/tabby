use std::collections::HashSet;

use anyhow::Result;
use async_trait::async_trait;
use juniper::GraphQLObject;
use validator::{validate_email, Validate, ValidationError};

#[async_trait]
pub trait SettingService: Send + Sync {
    async fn read_server_setting(&self) -> Result<ServerSetting>;
    async fn update_server_setting(&self, setting: ServerSetting) -> Result<()>;
}

#[derive(GraphQLObject, Debug, PartialEq)]
pub struct ServerSetting {
    pub security_allowed_register_domain_list: Vec<String>,
    pub security_disable_client_side_telemetry: bool,
    pub network_external_url: String,
}

#[derive(Validate)]
pub struct ServerSettingInput<'a> {
    #[validate(custom = "validate_unique_domains")]
    pub security_allowed_register_domain_list: Vec<&'a str>,
    #[validate(url)]
    pub network_external_url: &'a str,
}

fn validate_unique_domains(domains: &Vec<&str>) -> Result<(), ValidationError> {
    let unique: HashSet<_> = domains.iter().collect();
    if unique.len() != domains.len() {
        let collision = domains.iter().find(|s| unique.contains(s)).unwrap();
        let mut err = ValidationError::new("securityAllowedRegisterDomainList");
        err.message = Some(format!("Duplicate domain: {collision}").into());
        return Err(err);
    }
    for domain in domains {
        let email = format!("noreply@{domain}");
        if !validate_email(email) {
            let mut err = ValidationError::new("securityAllowedRegisterDomainList");
            err.message = Some(format!("Invalid domain name: {domain}").into());
            return Err(err);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::schema::setting::validate_unique_domains;

    #[test]
    fn test_validate_urls() {
        assert!(validate_unique_domains(&vec!["example.com"]).is_ok());

        assert!(validate_unique_domains(&vec!["https://example.com"]).is_err());

        assert!(validate_unique_domains(&vec!["domain.withmultipleparts.com"]).is_ok());

        assert!(validate_unique_domains(&vec!["example.com", "example.com"]).is_err());
    }
}
