use anyhow::Result;
use async_trait::async_trait;
use juniper::GraphQLObject;
use validator::{Validate, ValidationError};

#[async_trait]
pub trait SettingService: Send + Sync {
    async fn read_server_setting(&self) -> Result<ServerSetting>;
    async fn update_server_setting(&self, setting: ServerSetting) -> Result<()>;
}

#[derive(GraphQLObject)]
pub struct ServerSetting {
    pub security_allowed_register_domain_list: Vec<String>,
    pub security_disable_client_side_telemetry: bool,
    pub network_external_url: String,
}

#[derive(Validate)]
pub struct ServerSettingValidation<'a> {
    #[validate(custom = "validate_urls")]
    pub security_allowed_register_domain_list: Vec<&'a str>,
    #[validate(url)]
    pub network_external_url: &'a str,
}

fn validate_urls(urls: &Vec<&str>) -> Result<(), ValidationError> {
    for url in urls {
        if !validate_domain(*url) {
            return Err(ValidationError::new("invalid_url"));
        }
    }
    Ok(())
}

fn validate_domain(s: &str) -> bool {
    let Some(dot) = s.find('.') else { return false };
    let (site_name, tld) = s.split_at(dot);
    if !(0..=63).contains(&site_name.len())
        || site_name.starts_with("-")
        || site_name.ends_with("-")
    {
        return false;
    }
    let valid_site = site_name
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '-');
    let alphabetic_domains = tld
        .split('.')
        .flat_map(|s| s.chars())
        .all(|c| c.is_ascii_alphabetic());
    let domains_length = tld.split('.').all(|d| d.len() >= 2);

    valid_site && alphabetic_domains && domains_length
}
