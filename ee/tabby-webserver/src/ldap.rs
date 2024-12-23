use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use ldap3::{drive, result::Result, LdapConnAsync, Scope, SearchEntry};
use tabby_schema::auth::AuthenticationService;

#[async_trait]
pub trait LdapClient: Send + Sync {
    async fn validate(&mut self, user: &str, password: &str) -> Result<LdapUser>;
}

pub async fn new_ldap_client(auth: Arc<dyn AuthenticationService>) -> Arc<Mutex<dyn LdapClient>> {
    Arc::new(Mutex::new(LdapClientImpl { auth }))
}

pub struct LdapClientImpl {
    auth: Arc<dyn AuthenticationService>,
}

pub struct LdapUser {
    pub email: String,
    pub name: String,
}

#[async_trait]
impl LdapClient for LdapClientImpl {
    async fn validate(&mut self, user: &str, password: &str) -> Result<LdapUser> {
        let (connection, mut client) = LdapConnAsync::new("ldap://localhost:3890").await.unwrap();
        drive!(connection);

        // use bind_dn to search
        let res = client
            .simple_bind("cn=admin,ou=people,dc=ikw,dc=app", "password")
            .await?
            .success()?;
        println!("Bind successful {:?}", res);

        let searched = client
            .search(
                "dc=ikw,dc=app",
                Scope::OneLevel,
                format!("(uid={})", user).as_ref(),
                vec!["cn", "mail"],
            )
            .await?;

        println!("Search result {:?}", searched);

        if let Some(entry) = searched.0.into_iter().next() {
            let entry = SearchEntry::construct(entry);
            let user_dn = entry.dn;
            let email = entry
                .attrs
                .get("mail")
                .and_then(|v| v.get(0))
                .cloned()
                .unwrap_or_default();
            let name = entry
                .attrs
                .get("cn")
                .and_then(|v| v.get(0))
                .cloned()
                .unwrap_or_default();

            client.simple_bind(&user_dn, password).await?.success()?;

            println!("Search result, email {} name: {}", email, name);

            Ok(LdapUser { email, name })
        } else {
            Err(ldap3::LdapResult {
                rc: 32,
                matched: "".to_string(),
                text: "User not found".to_string(),
                refs: vec![],
                ctrls: vec![],
            }
            .into())
        }
    }
}

#[cfg(test)]
pub mod test_client {
    use super::*;
    use crate::service::FakeAuthService;

    #[tokio::test]
    async fn test_ldap_client() {
        let auth = FakeAuthService::new(vec![]);
        let client = new_ldap_client(Arc::new(auth)).await;
        client
            .lock()
            .unwrap()
            .validate("kw", "password")
            .await
            .unwrap();
    }
}
