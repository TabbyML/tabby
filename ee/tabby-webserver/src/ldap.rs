use anyhow::anyhow;
use async_trait::async_trait;
use ldap3::{drive, LdapConnAsync, LdapConnSettings, Scope, SearchEntry};
use tabby_schema::{CoreError, Result};

#[async_trait]
pub trait LdapClient: Send + Sync {
    async fn validate(&mut self, user: &str, password: &str) -> Result<LdapUser>;
}

pub fn new_ldap_client(
    host: &str,
    port: i64,
    encryption: &str,
    skip_verify_tls: bool,
    bind_dn: String,
    bind_password: &str,
    base_dn: String,
    user_filter: String,
    email_attr: String,
    name_attr: Option<String>,
) -> impl LdapClient {
    let mut settings = LdapConnSettings::new();
    if encryption == "starttls" {
        settings = settings.set_starttls(true);
    };
    if skip_verify_tls {
        settings = settings.set_no_tls_verify(true);
    };

    let schema = if encryption == "ldaps" {
        "ldaps"
    } else {
        "ldap"
    };

    LdapClientImpl {
        address: format!("{schema}://{host}:{port}"),
        bind_dn,
        bind_password: bind_password.to_string(),
        base_dn,
        user_filter,

        email_attr,
        name_attr,

        settings,
    }
}

pub struct LdapClientImpl {
    address: String,
    bind_dn: String,
    bind_password: String,
    base_dn: String,
    user_filter: String,

    email_attr: String,
    name_attr: Option<String>,

    settings: LdapConnSettings,
}

pub struct LdapUser {
    pub email: String,
    pub name: String,
}

#[async_trait]
impl LdapClient for LdapClientImpl {
    async fn validate(&mut self, user: &str, password: &str) -> Result<LdapUser> {
        let (connection, mut client) =
            LdapConnAsync::with_settings(self.settings.clone(), &self.address).await?;
        drive!(connection);

        // use bind_dn to search
        let _res = client
            .simple_bind(&self.bind_dn, &self.bind_password)
            .await?
            .success()?;

        let mut attrs = vec![&self.email_attr];
        if let Some(name_attr) = &self.name_attr {
            attrs.push(name_attr);
        }
        let searched = client
            .search(
                &self.base_dn,
                Scope::Subtree,
                &self.user_filter.replace("%s", user),
                attrs,
            )
            .await?;

        if let Some(entry) = searched.0.into_iter().next() {
            let entry = SearchEntry::construct(entry);
            let user_dn = entry.dn;
            let email = entry
                .attrs
                .get(&self.email_attr)
                .and_then(|v| v.first())
                .cloned()
                .ok_or_else(|| CoreError::Other(anyhow!("email not found for user")))?;
            let name = if let Some(name_attr) = &self.name_attr {
                entry
                    .attrs
                    .get(name_attr)
                    .and_then(|v| v.first())
                    .cloned()
                    .ok_or_else(|| CoreError::Other(anyhow!("name not found for user")))?
            } else {
                user.to_string()
            };

            client.simple_bind(&user_dn, password).await?.success()?;

            Ok(LdapUser { email, name })
        } else {
            Err(ldap3::LdapError::LdapResult {
                result: ldap3::LdapResult {
                    rc: 32,
                    matched: user.to_string(),
                    text: "User not found".to_string(),
                    refs: vec![],
                    ctrls: vec![],
                },
            }
            .into())
        }
    }
}
