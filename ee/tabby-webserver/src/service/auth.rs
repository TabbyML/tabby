use std::sync::Arc;

use anyhow::{anyhow, Context};
use argon2::{
    password_hash,
    password_hash::{rand_core::OsRng, SaltString},
    Argon2, PasswordHasher, PasswordVerifier,
};
use async_trait::async_trait;
use chrono::{Duration, Utc};
use juniper::ID;
use tabby_db::{DbConn, InvitationDAO};
use tabby_schema::{
    auth::{
        AuthenticationService, Invitation, JWTPayload, LdapCredential, OAuthCredential, OAuthError,
        OAuthProvider, OAuthResponse, RefreshTokenResponse, RegisterResponse,
        RequestInvitationInput, TokenAuthResponse, UpdateLdapCredentialInput,
        UpdateOAuthCredentialInput, UserSecured,
    },
    email::EmailService,
    is_demo_mode,
    license::{LicenseInfo, LicenseService},
    setting::SettingService,
    AsID, AsRowid, CoreError, DbEnum, Result,
};
use tokio::task::JoinHandle;
use tracing::warn;

use super::{graphql_pagination_to_filter, UserSecuredExt};
use crate::{
    bail,
    jwt::{generate_jwt, validate_jwt},
    ldap::{self, LdapClient},
    oauth::{self, OAuthClient},
};

#[cfg(test)]
pub mod testutils;

#[derive(Clone)]
struct ImpersonateUserCredential {
    id: i64,
    email: String,
    password_encrypted: String,
}

#[derive(Clone)]
struct AuthenticationServiceImpl {
    db: DbConn,
    mail: Arc<dyn EmailService>,
    license: Arc<dyn LicenseService>,
    setting: Arc<dyn SettingService>,
    impersonate_user: Option<ImpersonateUserCredential>,
}

pub fn create(
    db: DbConn,
    mail: Arc<dyn EmailService>,
    license: Arc<dyn LicenseService>,
    setting: Arc<dyn SettingService>,
) -> impl AuthenticationService {
    create_impl(db, mail, license, setting)
}

fn create_impl(
    db: DbConn,
    mail: Arc<dyn EmailService>,
    license: Arc<dyn LicenseService>,
    setting: Arc<dyn SettingService>,
) -> AuthenticationServiceImpl {
    let mut impersonate_user = None;
    if let Ok(value) = std::env::var("TABBY_OWNER_IMPERSONATE_OVERRIDE") {
        let words: Vec<&str> = value.split(':').collect();
        if words.len() == 2 {
            let password_encrypted = password_hash(words[1]).expect("can not encrypt password");
            impersonate_user = Some(ImpersonateUserCredential {
                // The first user registered is the owner user, so we set id = 1 to impersonate the owner user.
                id: 1,
                email: words[0].to_string(),
                password_encrypted,
            });
        }
    }
    AuthenticationServiceImpl {
        db,
        mail,
        license,
        setting,
        impersonate_user,
    }
}

#[async_trait]
impl AuthenticationService for AuthenticationServiceImpl {
    async fn register(
        &self,
        email: String,
        password: String,
        invitation_code: Option<String>,
        name: Option<String>,
    ) -> Result<RegisterResponse> {
        let is_admin_initialized = self.is_admin_initialized().await?;
        if is_admin_initialized && is_demo_mode() {
            bail!("Registering new users is disabled in demo mode");
        }
        let invitation =
            check_invitation(&self.db, is_admin_initialized, invitation_code, &email).await?;

        // check if email exists
        if self.db.get_user_by_email(&email).await?.is_some() {
            bail!("Email is already registered");
        }

        let Ok(pwd_hash) = password_hash(&password) else {
            bail!("Unknown error");
        };

        let id = if let Some(invitation) = invitation {
            self.db
                .create_user_with_invitation(
                    email.clone(),
                    Some(pwd_hash),
                    !is_admin_initialized,
                    invitation.id,
                    name.clone(),
                )
                .await?
        } else {
            self.db
                .create_user(
                    email.clone(),
                    Some(pwd_hash),
                    !is_admin_initialized,
                    name.clone(),
                )
                .await?
        };

        let refresh_token = self.db.create_refresh_token(id).await?;

        let Ok(access_token) = generate_jwt(id.as_id()) else {
            bail!("Unknown error");
        };

        if let Err(e) = self.mail.send_signup(email.clone()).await {
            warn!("Failed to send signup email: {e}");
        }

        let resp = RegisterResponse::new(access_token, refresh_token);
        Ok(resp)
    }

    async fn allow_self_signup(&self) -> Result<bool> {
        let domain_list = self
            .setting
            .read_security_setting()
            .await?
            .allowed_register_domain_list;
        let is_email_configured = self.mail.read_setting().await?.is_some();
        Ok(is_email_configured && !domain_list.is_empty())
    }

    async fn generate_reset_password_url(&self, id: &ID) -> Result<String> {
        let user = self.get_user(id).await?;
        if user.is_sso_user {
            bail!("Cannot generate reset password url for SSO users");
        }

        let external_url = self.setting.read_network_setting().await?.external_url;
        let id = id.as_rowid()?;
        let user = self.db.get_user(id).await?.context("User doesn't exits")?;
        if !user.active {
            bail!("Inactive user's password cannot be reset");
        }
        let code = self.db.create_password_reset(id).await?;
        let url = format!("{}/auth/reset-password?code={}", external_url, code);
        Ok(url)
    }

    async fn request_password_reset_email(&self, email: String) -> Result<Option<JoinHandle<()>>> {
        let user = self.get_user_by_email(&email).await.ok();

        let Some(user @ UserSecured { active: true, .. }) = user else {
            return Ok(None);
        };

        if user.is_sso_user {
            bail!("Cannot request password reset for SSO users");
        }

        let id = user.id.as_rowid()?;

        // request_password_reset_email is invoked by the user, so we need to check for existing password reset requests to prevent spamming
        let existing = self.db.get_password_reset_by_user_id(id).await?;
        if let Some(existing) = existing {
            if Utc::now().signed_duration_since(existing.created_at) < Duration::minutes(5) {
                bail!("A password reset has been requested recently, please try again later");
            }
        }
        let code = self.db.create_password_reset(id).await?;
        let handle = self
            .mail
            .send_password_reset(user.email, code.clone())
            .await?;
        Ok(Some(handle))
    }

    async fn password_reset(&self, code: &str, password: &str) -> Result<()> {
        let password_encrypted = password_hash(password).map_err(|_| anyhow!("Unknown error"))?;

        let user_id = self.db.verify_password_reset(code).await?;
        let user = self.get_user(&user_id.as_id()).await?;
        if user.is_sso_user {
            bail!("Password cannot be reset for SSO users");
        }

        let old_pass_encrypted = self
            .db
            .get_user(user_id)
            .await?
            .expect("User must exist")
            .password_encrypted;

        if old_pass_encrypted.is_some_and(|old| password_verify(password, &old)) {
            bail!("New password cannot be the same as your current password");
        }
        self.db.delete_password_reset_by_user_id(user_id).await?;
        self.db
            .update_user_password(user_id, password_encrypted)
            .await?;
        Ok(())
    }

    async fn update_user_password(
        &self,
        id: &ID,
        old_password: Option<&str>,
        new_password: &str,
    ) -> Result<()> {
        if is_demo_mode() {
            bail!("Changing passwords is disabled in demo mode");
        }

        let user = self.get_user(id).await?;
        if user.is_sso_user {
            bail!("Password cannot be changed for SSO users");
        }

        let user = self
            .db
            .get_user(id.as_rowid()?)
            .await?
            .ok_or_else(|| anyhow!("Invalid user"))?;

        let password_verified = match (user.password_encrypted, old_password) {
            // If the user had no password previously and specified no new password, that's fine
            (None, None) => true,
            // If they had a previous password, they must specify what it was and it must match
            (Some(user_password), Some(old_password)) => {
                password_verify(old_password, &user_password)
            }
            // Otherwise, they must have either specified a password when they did not have one
            // or failed to specify a password when they did have one
            _ => false,
        };
        if !password_verified {
            bail!("Password is incorrect");
        }

        if old_password.is_some_and(|pass| pass == new_password) {
            bail!("New password cannot be the same as your current password");
        }

        let new_password_encrypted =
            password_hash(new_password).map_err(|_| anyhow!("Unknown error"))?;
        self.db
            .update_user_password(user.id, new_password_encrypted)
            .await?;
        Ok(())
    }

    async fn update_user_avatar(&self, id: &ID, avatar: Option<Box<[u8]>>) -> Result<()> {
        if is_demo_mode() {
            bail!("Changing profile data is disabled in demo mode");
        }
        if avatar.as_ref().is_some_and(|v| v.len() > 512 * 1024) {
            bail!("The image you are attempting to upload is too large. Please ensure the file size is under 512KB");
        }
        let id = id.as_rowid()?;
        self.db.update_user_avatar(id, avatar).await?;
        Ok(())
    }

    async fn get_user_avatar(&self, id: &ID) -> Result<Option<Box<[u8]>>> {
        Ok(self.db.get_user_avatar(id.as_rowid()?).await?)
    }

    async fn update_user_name(&self, id: &ID, name: String) -> Result<()> {
        if is_demo_mode() {
            bail!("Changing profile data is disabled in demo mode");
        }

        let user = self.get_user(id).await?;
        if user.is_sso_user {
            bail!("Name cannot be changed for SSO users");
        }

        let id = id.as_rowid()?;
        self.db.update_user_name(id, name).await?;
        Ok(())
    }

    async fn token_auth(&self, email: String, password: String) -> Result<TokenAuthResponse> {
        if let Some(user) = &self.impersonate_user {
            if user.email == email && password_verify(&password, &user.password_encrypted) {
                let refresh_token = self.db.create_refresh_token(user.id).await?;
                let Ok(access_token) = generate_jwt(user.id.as_id()) else {
                    bail!("Unknown error");
                };
                let resp = TokenAuthResponse::new(access_token, refresh_token);
                return Ok(resp);
            }
        }

        let Some(user) = self.db.get_user_by_email(&email).await? else {
            bail!("Invalid email address or password");
        };

        if !user
            .password_encrypted
            .is_some_and(|pass| password_verify(&password, &pass))
        {
            bail!("Invalid email address or password");
        }

        if !user.active {
            bail!("User is disabled");
        }

        let refresh_token = self.db.create_refresh_token(user.id).await?;

        let Ok(access_token) = generate_jwt(user.id.as_id()) else {
            bail!("Unknown error");
        };

        let resp = TokenAuthResponse::new(access_token, refresh_token);
        Ok(resp)
    }

    async fn token_auth_ldap(&self, user_id: &str, password: &str) -> Result<TokenAuthResponse> {
        let license = self
            .license
            .read()
            .await
            .context("Failed to read license info")?;

        let credential = self.db.read_ldap_credential().await?;
        if credential.is_none() {
            bail!("LDAP is not configured");
        }

        let credential = credential.unwrap();
        let mut client = ldap::new_ldap_client(
            credential.host.as_ref(),
            credential.port,
            credential.encryption.as_str(),
            credential.skip_tls_verify,
            credential.bind_dn,
            &credential.bind_password,
            credential.base_dn,
            credential.user_filter,
            credential.email_attribute,
            credential.name_attribute,
        );

        ldap_login(
            &mut client,
            &self.db,
            &*self.setting,
            &license,
            &*self.mail,
            user_id,
            password,
        )
        .await
    }

    async fn refresh_token(&self, token: String) -> Result<RefreshTokenResponse> {
        let Some(refresh_token) = self.db.get_refresh_token(&token).await? else {
            bail!("Invalid refresh token");
        };
        if refresh_token.is_expired() {
            bail!("Expired refresh token");
        }
        let Some(user) = self.db.get_user(refresh_token.user_id).await? else {
            bail!("User not found");
        };

        if !user.active {
            bail!("User is disabled");
        }

        let new_token = self
            .db
            .renew_refresh_token(refresh_token.id, &token)
            .await?;

        // refresh token update is done, generate new access token based on user info
        let Ok(access_token) = generate_jwt(user.id.as_id()) else {
            bail!("Unknown error");
        };

        let resp = RefreshTokenResponse::new(access_token, new_token, refresh_token.expires_at);

        Ok(resp)
    }

    async fn verify_access_token(&self, access_token: &str) -> Result<JWTPayload> {
        let claims = validate_jwt(access_token).map_err(anyhow::Error::new)?;
        Ok(claims)
    }
    async fn verify_auth_token(&self, token: &str) -> Result<ID> {
        match self.db.verify_auth_token(token, false).await {
            Ok(user) => Ok(user.as_id()),
            Err(e) => bail!("Failed to verify auth token: {e}"),
        }
    }

    async fn is_admin_initialized(&self) -> Result<bool> {
        let admin = self.db.list_admin_users().await?;
        Ok(!admin.is_empty())
    }

    async fn update_user_role(&self, id: &ID, is_admin: bool) -> Result<()> {
        if is_admin {
            let license = self.license.read().await?;
            let num_admins = self.db.count_active_admin_users().await?;
            license.ensure_admin_seats(num_admins + 1)?;
        }

        let id = id.as_rowid()?;
        let user = self.db.get_user(id).await?.context("User doesn't exits")?;

        if !user.active {
            bail!("Inactive user's status cannot be changed");
        }

        if user.is_owner() {
            bail!("The owner's admin status cannot be changed");
        }

        Ok(self.db.update_user_role(id, is_admin).await?)
    }

    async fn get_user_by_email(&self, email: &str) -> Result<UserSecured> {
        let user = self.db.get_user_by_email(email).await?;
        if let Some(dao) = user {
            Ok(UserSecured::new(self.db.clone(), dao))
        } else {
            bail!("User not found {}", email)
        }
    }

    async fn get_user(&self, id: &ID) -> Result<UserSecured> {
        let user = self.db.get_user(id.as_rowid()?).await?;
        if let Some(dao) = user {
            Ok(UserSecured::new(self.db.clone(), dao))
        } else {
            bail!("User not found")
        }
    }

    async fn create_invitation(&self, email: String) -> Result<Invitation> {
        if is_demo_mode() {
            bail!("Inviting users is disabled in demo mode");
        }
        let license = self.license.read().await?;
        license.ensure_available_seats(1)?;

        let invitation = self.db.create_invitation(email.clone()).await?;
        let email_sent = self
            .mail
            .send_invitation(email, invitation.code.clone())
            .await;
        match email_sent {
            Ok(_) | Err(CoreError::EmailNotConfigured) => {}
            Err(e) => warn!(
                "Failed to send invitation email, please check your SMTP settings are correct: {e}"
            ),
        }
        Ok(invitation.into())
    }

    async fn request_invitation_email(&self, input: RequestInvitationInput) -> Result<Invitation> {
        if !self
            .setting
            .read_security_setting()
            .await?
            .can_register_without_invitation(&input.email)
        {
            bail!("Your email does not belong to any known authentication domains. Please contact the administrator for assistance.");
        }
        let invitation = AuthenticationService::create_invitation(self, input.email).await?;
        Ok(invitation)
    }

    async fn delete_invitation(&self, id: &ID) -> Result<ID> {
        Ok((self.db.delete_invitation(id.as_rowid()?).await?).as_id())
    }

    async fn reset_user_auth_token(&self, id: &ID) -> Result<()> {
        Ok(self.db.reset_user_auth_token_by_id(id.as_rowid()?).await?)
    }

    async fn logout_all_sessions(&self, id: &ID) -> Result<()> {
        Ok(self.db.delete_tokens_by_user_id(id.as_rowid()?).await?)
    }

    async fn list_users(
        &self,
        ids: Option<Vec<ID>>,
        after: Option<String>,
        before: Option<String>,
        first: Option<usize>,
        last: Option<usize>,
    ) -> Result<Vec<UserSecured>> {
        let (skip_id, limit, backwards) = graphql_pagination_to_filter(after, before, first, last)?;

        let rowids = ids.map(|ids| {
            ids.into_iter()
                .filter_map(|x| x.as_rowid().ok().map(|x| x as i32))
                .collect()
        });

        Ok(self
            .db
            .list_users_with_filter(rowids, skip_id, limit, backwards)
            .await?
            .into_iter()
            .map(|x| UserSecured::new(self.db.clone(), x))
            .collect())
    }

    async fn list_invitations(
        &self,
        after: Option<String>,
        before: Option<String>,
        first: Option<usize>,
        last: Option<usize>,
    ) -> Result<Vec<Invitation>> {
        let (limit, skip_id, backwards) = graphql_pagination_to_filter(after, before, first, last)?;
        Ok(self
            .db
            .list_invitations_with_filter(limit, skip_id, backwards)
            .await?
            .into_iter()
            .map(|x| x.into())
            .collect())
    }

    async fn oauth(
        &self,
        code: String,
        provider: OAuthProvider,
    ) -> std::result::Result<OAuthResponse, OAuthError> {
        let client = oauth::new_oauth_client(provider, Arc::new(self.clone()));
        let license = self
            .license
            .read()
            .await
            .context("Failed to read license info")?;

        oauth_login(
            client,
            code,
            &self.db,
            &*self.setting,
            &license,
            &*self.mail,
        )
        .await
    }

    async fn read_oauth_credential(
        &self,
        provider: OAuthProvider,
    ) -> Result<Option<OAuthCredential>> {
        let credential = self
            .db
            .read_oauth_credential(provider.as_enum_str())
            .await?;
        match credential {
            Some(c) => Ok(Some(c.try_into()?)),
            None => Ok(None),
        }
    }

    async fn oauth_callback_url(&self, provider: OAuthProvider) -> Result<String> {
        let external_url = self.setting.read_network_setting().await?.external_url;
        let url = match provider {
            OAuthProvider::Github => external_url + "/oauth/callback/github",
            OAuthProvider::Google => external_url + "/oauth/callback/google",
            OAuthProvider::Gitlab => external_url + "/oauth/callback/gitlab",
        };
        Ok(url)
    }

    async fn update_oauth_credential(&self, input: UpdateOAuthCredentialInput) -> Result<()> {
        self.db
            .update_oauth_credential(
                input.provider.as_enum_str(),
                &input.client_id,
                input.client_secret.as_deref(),
            )
            .await?;
        Ok(())
    }

    async fn delete_oauth_credential(&self, provider: OAuthProvider) -> Result<()> {
        self.db
            .delete_oauth_credential(provider.as_enum_str())
            .await?;
        Ok(())
    }

    async fn read_ldap_credential(&self) -> Result<Option<LdapCredential>> {
        let credential = self.db.read_ldap_credential().await?;
        match credential {
            Some(c) => Ok(Some(c.try_into()?)),
            None => Ok(None),
        }
    }

    async fn test_ldap_connection(&self, input: UpdateLdapCredentialInput) -> Result<()> {
        let password = if let Some(password) = input.bind_password.as_deref() {
            password
        } else {
            &self
                .db
                .read_ldap_credential()
                .await?
                .ok_or_else(|| anyhow!("LDAP password is not configured"))?
                .bind_password
        };
        let mut client = ldap::new_ldap_client(
            input.host.as_ref(),
            input.port as i64,
            input.encryption.as_enum_str(),
            input.skip_tls_verify,
            input.bind_dn,
            password,
            input.base_dn,
            input.user_filter,
            input.email_attribute,
            input.name_attribute,
        );

        if let Err(e) = client.validate("", "").await {
            if e.to_string().contains("User not found") {
                return Ok(());
            } else {
                bail!("Failed to connect to LDAP server: {e}");
            }
        }

        Ok(())
    }

    async fn update_ldap_credential(&self, input: UpdateLdapCredentialInput) -> Result<()> {
        let password = if let Some(password) = input.bind_password.as_deref() {
            password
        } else {
            &self
                .db
                .read_ldap_credential()
                .await?
                .ok_or_else(|| anyhow!("LDAP password is not configured"))?
                .bind_password
        };
        self.db
            .update_ldap_credential(
                &input.host,
                input.port,
                &input.bind_dn,
                password,
                &input.base_dn,
                &input.user_filter,
                input.encryption.as_enum_str(),
                input.skip_tls_verify,
                &input.email_attribute,
                input.name_attribute.as_deref(),
            )
            .await?;
        Ok(())
    }

    async fn delete_ldap_credential(&self) -> Result<()> {
        self.db.delete_ldap_credential().await?;
        Ok(())
    }

    async fn update_user_active(&self, id: &ID, active: bool) -> Result<()> {
        let id = id.as_rowid()?;
        let user = self.db.get_user(id).await?.context("User doesn't exits")?;

        if user.active == active {
            bail!("User's active status is already set to {active}");
        }

        if user.is_owner() {
            bail!("The owner's active status cannot be changed");
        }

        let license = self.license.read().await?;

        if active {
            // Check there's sufficient seat if switching user to active.
            license.ensure_available_seats(1)?;
        }

        if active && user.is_admin {
            // Check there's sufficient seat if an admin being swtiched to active.
            let num_admins = self.db.count_active_admin_users().await?;
            license.ensure_admin_seats(num_admins + 1)?;
        }

        Ok(self.db.update_user_active(id, active).await?)
    }
}

async fn ldap_login(
    client: &mut dyn LdapClient,
    db: &DbConn,
    setting: &dyn SettingService,
    license: &LicenseInfo,
    mail: &dyn EmailService,
    user_id: &str,
    password: &str,
) -> Result<TokenAuthResponse> {
    let user = client.validate(user_id, password).await?;
    let user_id = get_or_create_sso_user(license, db, setting, mail, &user.email, &user.name)
        .await
        .map_err(|e| CoreError::Other(anyhow!("fail to get or create ldap user: {}", e)))?;

    let refresh_token = db.create_refresh_token(user_id).await?;
    let access_token = generate_jwt(user_id.as_id())
        .map_err(|e| CoreError::Other(anyhow!("fail to create access_token: {}", e)))?;

    let resp = TokenAuthResponse::new(access_token, refresh_token);
    Ok(resp)
}

async fn oauth_login(
    client: Arc<dyn OAuthClient>,
    code: String,
    db: &DbConn,
    setting: &dyn SettingService,
    license: &LicenseInfo,
    mail: &dyn EmailService,
) -> Result<OAuthResponse, OAuthError> {
    let access_token = client.exchange_code_for_token(code).await?;
    let email = client.fetch_user_email(&access_token).await?;
    let name = client.fetch_user_full_name(&access_token).await?;
    let user_id = get_or_create_sso_user(license, db, setting, mail, &email, &name).await?;

    let refresh_token = db.create_refresh_token(user_id).await?;

    let access_token = generate_jwt(user_id.as_id()).map_err(|_| OAuthError::Unknown)?;

    let resp = OAuthResponse {
        access_token,
        refresh_token,
    };
    Ok(resp)
}

async fn get_or_create_sso_user(
    license: &LicenseInfo,
    db: &DbConn,
    setting: &dyn SettingService,
    mail: &dyn EmailService,
    email: &str,
    name: &str,
) -> Result<i64, OAuthError> {
    if let Some(user) = db.get_user_by_email(email).await? {
        return user
            .active
            .then_some(user.id)
            .ok_or(OAuthError::UserDisabled);
    }

    // Check license before creating user.
    if license.ensure_available_seats(1).is_err() {
        return Err(OAuthError::InsufficientSeats);
    }

    let name = (!name.is_empty()).then_some(name.to_owned());

    if setting
        .read_security_setting()
        .await
        .map_err(|x| OAuthError::Other(x.into()))?
        .can_register_without_invitation(email)
    {
        if is_demo_mode() {
            bail!("Registering new users is disabled in demo mode");
        }
        // it's ok to set password to null here, because
        // 1. both `register` & `token_auth` mutation will do input validation, so empty password won't be accepted
        // 2. `password_verify` will always return false for empty password hash read from user table
        // so user created here is only able to login by github oauth, or ldap, normal login won't work

        let res = db.create_user(email.to_owned(), None, false, name).await?;
        if let Err(e) = mail.send_signup(email.to_string()).await {
            warn!("Failed to send signup email: {e}");
        }
        Ok(res)
    } else {
        let Some(invitation) = db.get_invitation_by_email(email).await.ok().flatten() else {
            return Err(OAuthError::UserNotInvited);
        };
        // safe to create with empty password for same reasons above
        let id = db
            .create_user_with_invitation(email.to_owned(), None, false, invitation.id, name)
            .await?;
        let user = db.get_user(id).await?.unwrap();
        Ok(user.id)
    }
}

async fn check_invitation(
    db: &DbConn,
    is_admin_initialized: bool,
    invitation_code: Option<String>,
    email: &str,
) -> Result<Option<InvitationDAO>> {
    if !is_admin_initialized {
        // Creating the admin user, no invitation required
        return Ok(None);
    }

    let err = Err(anyhow!("Invitation code is not valid").into());
    let Some(invitation_code) = invitation_code else {
        return err;
    };

    let Some(invitation) = db.get_invitation_by_code(&invitation_code).await? else {
        return err;
    };

    if invitation.email != email {
        bail!("Invitation code is not for this email address");
    }

    Ok(Some(invitation))
}

fn password_hash(raw: &str) -> password_hash::Result<String> {
    let salt = SaltString::generate(&mut OsRng);
    let argon2 = Argon2::default();
    let hash = argon2.hash_password(raw.as_bytes(), &salt)?.to_string();

    Ok(hash)
}

fn password_verify(raw: &str, hash: &str) -> bool {
    if let Ok(parsed_hash) = argon2::PasswordHash::new(hash) {
        let argon2 = Argon2::default();
        argon2.verify_password(raw.as_bytes(), &parsed_hash).is_ok()
    } else {
        false
    }
}

#[cfg(test)]
mod tests {
    use tabby_schema::auth::LdapEncryptionKind;

    use crate::service::auth::testutils::FakeLdapClient;

    struct MockLicenseService {
        status: LicenseStatus,
        seats: i32,
        seats_used: i32,
    }

    impl MockLicenseService {
        fn team() -> Self {
            Self {
                status: LicenseStatus::Ok,
                seats: 5,
                seats_used: 1,
            }
        }

        fn team_with_seats(seats: i32) -> Self {
            Self {
                status: LicenseStatus::Ok,
                seats,
                seats_used: 1,
            }
        }

        fn invalid() -> Self {
            Self {
                status: LicenseStatus::Expired,
                seats: 5,
                seats_used: 1,
            }
        }
    }

    #[async_trait]
    impl LicenseService for MockLicenseService {
        async fn read(&self) -> Result<LicenseInfo> {
            Ok(LicenseInfo {
                r#type: tabby_schema::license::LicenseType::Team,
                status: self.status.clone(),
                seats: self.seats,
                seats_used: self.seats_used,
                issued_at: Some(Utc::now()),
                expires_at: Some(Utc::now()),
            })
        }

        async fn update(&self, _: String) -> Result<()> {
            unimplemented!()
        }

        async fn reset(&self) -> Result<()> {
            unimplemented!()
        }
    }

    async fn test_authentication_service_with_license(
        license: Arc<dyn LicenseService>,
    ) -> AuthenticationServiceImpl {
        let db = DbConn::new_in_memory().await.unwrap();
        create_impl(
            db.clone(),
            Arc::new(new_email_service(db.clone()).await.unwrap()),
            license,
            Arc::new(crate::service::setting::create(db)),
        )
    }

    async fn test_authentication_service() -> AuthenticationServiceImpl {
        test_authentication_service_with_license(Arc::new(MockLicenseService::team())).await
    }

    async fn test_authentication_service_without_valid_license() -> AuthenticationServiceImpl {
        test_authentication_service_with_license(Arc::new(MockLicenseService::invalid())).await
    }

    async fn test_authentication_service_with_mail() -> (AuthenticationServiceImpl, TestEmailServer)
    {
        let db = DbConn::new_in_memory().await.unwrap();
        let smtp = TestEmailServer::start().await;
        let service = AuthenticationServiceImpl {
            db: db.clone(),
            mail: Arc::new(smtp.create_test_email_service(db.clone()).await),
            license: Arc::new(MockLicenseService::team()),
            setting: Arc::new(crate::service::setting::create(db)),
            impersonate_user: None,
        };
        (service, smtp)
    }

    use assert_matches::assert_matches;
    use serial_test::serial;
    use tabby_schema::{
        juniper::relay::{self, Connection},
        license::{LicenseInfo, LicenseStatus, LicenseType},
    };

    use super::*;
    use crate::{
        oauth::test_client::TestOAuthClient,
        service::email::{new_email_service, testutils::TestEmailServer},
    };

    #[test]
    fn test_password_hash() {
        let raw = "12345678";
        let hash = password_hash(raw).unwrap();

        assert_eq!(hash.len(), 97);
        assert!(hash.starts_with("$argon2id$v=19$m=19456,t=2,p=1$"));
    }

    #[test]
    fn test_password_verify() {
        let raw = "12345678";
        let hash = password_hash(raw).unwrap();

        assert!(password_verify(raw, &hash));
        assert!(!password_verify(raw, "invalid hash"));
    }

    static ADMIN_EMAIL: &str = "test@example.com";
    static ADMIN_PASSWORD: &str = "123456789$acR";

    async fn register_admin_user(service: &impl AuthenticationService) -> RegisterResponse {
        service
            .register(
                ADMIN_EMAIL.to_owned(),
                ADMIN_PASSWORD.to_owned(),
                None,
                None,
            )
            .await
            .unwrap()
    }

    #[tokio::test]
    async fn test_auth_token() {
        let service = test_authentication_service().await;
        assert_matches!(
            service
                .token_auth(ADMIN_EMAIL.to_owned(), "12345678".to_owned())
                .await,
            Err(_)
        );

        register_admin_user(&service).await;

        assert_matches!(
            service
                .token_auth(ADMIN_EMAIL.to_owned(), "12345678".to_owned())
                .await,
            Err(_)
        );

        let resp1 = service
            .token_auth(ADMIN_EMAIL.to_owned(), ADMIN_PASSWORD.to_owned())
            .await
            .unwrap();
        let resp2 = service
            .token_auth(ADMIN_EMAIL.to_owned(), ADMIN_PASSWORD.to_owned())
            .await
            .unwrap();
        // each auth should generate a new refresh token
        assert_ne!(resp1.refresh_token, resp2.refresh_token);
    }

    #[tokio::test]
    async fn test_invitation_flow() {
        let service = test_authentication_service().await;

        assert!(!service.is_admin_initialized().await.unwrap());
        register_admin_user(&service).await;

        let email = "user@user.com";
        let password = "12345678dD^";

        service.create_invitation(email.to_owned()).await.unwrap();
        let invitation = &service
            .list_invitations(None, None, None, None)
            .await
            .unwrap()[0];

        // Admin initialized, registeration requires a invitation code;
        assert_matches!(
            service
                .register(email.to_owned(), password.to_owned(), None, None)
                .await,
            Err(_)
        );

        // Invalid invitation code won't work.
        assert_matches!(
            service
                .register(
                    email.to_owned(),
                    password.to_owned(),
                    Some("abc".to_owned()),
                    None
                )
                .await,
            Err(_)
        );

        // Register success.
        assert!(service
            .register(
                email.to_owned(),
                password.to_owned(),
                Some(invitation.code.clone()),
                None
            )
            .await
            .is_ok());

        // Try register again with same email failed.
        assert_matches!(
            service
                .register(
                    email.to_owned(),
                    password.to_owned(),
                    Some(invitation.code.clone()),
                    None
                )
                .await,
            Err(_)
        );

        // Used invitation should have been deleted,  following delete attempt should fail.
        assert!(service
            .db
            .delete_invitation(invitation.id.as_rowid().unwrap() as i64)
            .await
            .is_err());
    }

    #[tokio::test]
    async fn test_refresh_token() {
        let service = test_authentication_service().await;
        let reg = register_admin_user(&service).await;

        let resp1 = service
            .refresh_token(reg.refresh_token.clone())
            .await
            .unwrap();
        // new access token should be valid
        assert!(validate_jwt(&resp1.access_token).is_ok());
        // refresh token should be renewed
        assert_ne!(reg.refresh_token, resp1.refresh_token);

        let resp2 = service
            .refresh_token(resp1.refresh_token.clone())
            .await
            .unwrap();
        // expire time should be no change
        assert_eq!(resp1.refresh_expires_at, resp2.refresh_expires_at);
    }

    #[tokio::test]
    async fn test_reset_user_auth_token() {
        let service = test_authentication_service().await;
        register_admin_user(&service).await;

        let user = service.get_user_by_email(ADMIN_EMAIL).await.unwrap();
        service.reset_user_auth_token(&user.id).await.unwrap();

        let user2 = service.get_user_by_email(ADMIN_EMAIL).await.unwrap();
        assert_ne!(user.auth_token, user2.auth_token);
    }

    #[tokio::test]
    async fn test_is_admin_initialized() {
        let service = test_authentication_service().await;

        assert!(!service.is_admin_initialized().await.unwrap());
        tabby_db::testutils::create_user(&service.db).await;
        assert!(service.is_admin_initialized().await.unwrap());
    }

    async fn list_users(
        db: &AuthenticationServiceImpl,
        after: Option<String>,
        before: Option<String>,
        first: Option<i32>,
        last: Option<i32>,
    ) -> Connection<UserSecured> {
        relay::query_async(
            after,
            before,
            first,
            last,
            |after, before, first, last| async move {
                Ok(db
                    .list_users(None, after, before, first, last)
                    .await
                    .unwrap())
            },
        )
        .await
        .unwrap()
    }

    #[tokio::test]
    async fn test_request_invitation() {
        let service = test_authentication_service().await;
        service
            .db
            .update_security_setting(Some("example.com".into()), false)
            .await
            .unwrap();

        assert!(service
            .request_invitation_email(RequestInvitationInput {
                email: "test@example.com".into()
            })
            .await
            .is_ok());

        assert!(service
            .request_invitation_email(RequestInvitationInput {
                email: "test@gmail.com".into()
            })
            .await
            .is_err());
    }

    #[tokio::test]
    #[serial]
    async fn test_get_or_create_oauth_user() {
        let (service, mail) = test_authentication_service_with_mail().await;
        let license = service.license.read().await.unwrap();
        let id = service
            .db
            .create_user("test@example.com".into(), None, false, None)
            .await
            .unwrap();
        service.db.update_user_active(id, false).await.unwrap();
        let setting = service.setting;

        let res = get_or_create_sso_user(
            &license,
            &service.db,
            &*setting,
            &*service.mail,
            "test@example.com",
            "",
        )
        .await;
        assert_matches!(res, Err(OAuthError::UserDisabled));

        service
            .db
            .update_security_setting(Some("example.com".into()), false)
            .await
            .unwrap();

        let res = get_or_create_sso_user(
            &license,
            &service.db,
            &*setting,
            &*service.mail,
            "example@example.com",
            "Example User",
        )
        .await;
        assert_matches!(res, Ok(2));

        let user = service.db.get_user(2).await.unwrap().unwrap();
        assert_eq!(user.email, "example@example.com");
        assert_eq!(user.name, Some("Example User".into()));

        tokio::time::sleep(Duration::milliseconds(50).to_std().unwrap()).await;
        assert_eq!(mail.list_mail().await[0].subject, "Welcome to Tabby!");

        let res = get_or_create_sso_user(
            &license,
            &service.db,
            &*setting,
            &*service.mail,
            "example@gmail.com",
            "",
        )
        .await;
        assert_matches!(res, Err(OAuthError::UserNotInvited));

        service
            .db
            .create_invitation("example@gmail.com".into())
            .await
            .unwrap();

        let res = get_or_create_sso_user(
            &license,
            &service.db,
            &*setting,
            &*service.mail,
            "example@gmail.com",
            "User 3 by Invitation",
        )
        .await;
        assert_matches!(res, Ok(3));

        let user = service.db.get_user(3).await.unwrap().unwrap();
        assert_eq!(user.email, "example@gmail.com");
        assert_eq!(user.name, Some("User 3 by Invitation".into()));
    }

    #[tokio::test]
    #[serial]
    async fn test_register_email() {
        let (service, mail) = test_authentication_service_with_mail().await;
        let code = service
            .db
            .create_invitation("test@example.com".into())
            .await
            .unwrap();

        service
            .register("test@example.com".into(), "".into(), Some(code.code), None)
            .await
            .unwrap();

        tokio::time::sleep(Duration::milliseconds(50).to_std().unwrap()).await;

        assert_eq!(mail.list_mail().await[0].subject, "Welcome to Tabby!");
    }

    #[tokio::test]
    async fn test_update_role() {
        let service = test_authentication_service().await;
        let _ = service
            .db
            .create_user("admin@example.com".into(), None, true, None)
            .await
            .unwrap();

        let user_id = service
            .db
            .create_user("user@example.com".into(), None, false, None)
            .await
            .unwrap();

        assert!(service
            .update_user_role(&user_id.as_id(), true)
            .await
            .is_ok());

        // Inactive user's role cannot be changed
        service
            .update_user_active(&user_id.as_id(), false)
            .await
            .unwrap();
        assert!(service
            .update_user_role(&user_id.as_id(), false)
            .await
            .is_err());
    }

    #[tokio::test]
    async fn test_owner_status() {
        let service = test_authentication_service().await;
        let admin_id = service
            .db
            .create_user("admin@example.com".into(), None, true, None)
            .await
            .unwrap();

        assert!(service
            .update_user_role(&admin_id.as_id(), false)
            .await
            .is_err());

        assert!(service
            .update_user_active(&admin_id.as_id(), false)
            .await
            .is_err());
    }

    #[tokio::test]
    #[serial]
    async fn test_password_reset() {
        let (service, smtp) = test_authentication_service_with_mail().await;

        // Test first reset, ensure wrong code fails
        service
            .db
            .create_user("user@example.com".into(), Some("pass".into()), true, None)
            .await
            .unwrap();
        let user = service.get_user_by_email("user@example.com").await.unwrap();

        let handle = service
            .request_password_reset_email("user@example.com".into())
            .await
            .unwrap();
        handle.unwrap().await.unwrap();
        assert!(smtp.list_mail().await[0]
            .subject
            .to_lowercase()
            .contains("password"));

        let reset = service
            .db
            .get_password_reset_by_user_id(user.id.as_rowid().unwrap())
            .await
            .unwrap()
            .unwrap();

        assert!(service.password_reset("", "newpass").await.is_err());
        assert!(service.password_reset(&reset.code, "newpass").await.is_ok());

        // Test second reset, ensure expired code fails
        let user = service
            .db
            .get_user_by_email(&user.email)
            .await
            .unwrap()
            .unwrap();
        assert_ne!(user.password_encrypted, Some("pass".into()));

        service
            .request_password_reset_email("user@example.com".into())
            .await
            .unwrap();
        let reset = service
            .db
            .get_password_reset_by_user_id(user.id)
            .await
            .unwrap()
            .unwrap();

        service
            .db
            .mark_password_reset_expired(&reset.code)
            .await
            .unwrap();

        assert!(service
            .password_reset(&reset.code, "newpass2")
            .await
            .is_err());

        // Test third reset, ensure inactive users cannot reset their password
        let user_id_2 = service
            .db
            .create_user("user2@example.com".into(), Some("pass".into()), false, None)
            .await
            .unwrap();

        service
            .request_password_reset_email("user2@example.com".into())
            .await
            .unwrap();
        let reset = service
            .db
            .get_password_reset_by_user_id(user_id_2)
            .await
            .unwrap()
            .unwrap();

        service
            .db
            .update_user_active(user_id_2, false)
            .await
            .unwrap();

        assert!(service
            .password_reset(&reset.code, "newpass")
            .await
            .is_err());
    }

    #[tokio::test]
    async fn test_pagination() {
        let service = test_authentication_service().await;
        service
            .db
            .create_user("a@example.com".into(), Some("pass".into()), false, None)
            .await
            .unwrap();
        service
            .db
            .create_user("b@example.com".into(), Some("pass".into()), false, None)
            .await
            .unwrap();
        service
            .db
            .create_user("c@example.com".into(), Some("pass".into()), false, None)
            .await
            .unwrap();

        let all_users = list_users(&service, None, None, None, None).await;

        assert!(!all_users.page_info.has_next_page);
        assert!(!all_users.page_info.has_previous_page);

        let users = list_users(
            &service,
            Some(all_users.edges[0].cursor.clone()),
            None,
            None,
            None,
        )
        .await;

        assert!(!users.page_info.has_next_page);
        assert!(users.page_info.has_previous_page);

        let users = list_users(&service, None, None, Some(2), None).await;

        assert!(users.page_info.has_next_page);
        assert!(!users.page_info.has_previous_page);

        let users = list_users(
            &service,
            None,
            Some(all_users.edges[1].cursor.clone()),
            None,
            Some(1),
        )
        .await;

        assert!(users.page_info.has_next_page);
        assert!(!users.page_info.has_previous_page);

        let users = list_users(
            &service,
            Some(all_users.edges[2].cursor.clone()),
            None,
            None,
            None,
        )
        .await;
        assert!(!users.page_info.has_next_page);
        assert!(users.page_info.has_previous_page);

        let users = list_users(&service, None, None, Some(3), None).await;
        assert!(!users.page_info.has_next_page);
        assert!(!users.page_info.has_previous_page);

        let users = list_users(
            &service,
            Some(all_users.edges[0].cursor.clone()),
            None,
            Some(2),
            None,
        )
        .await;
        assert!(!users.page_info.has_next_page);
        assert!(users.page_info.has_previous_page);
    }

    #[tokio::test]
    #[serial]
    async fn test_allow_self_signup() {
        let (service, _) = test_authentication_service_with_mail().await;

        assert!(!service.allow_self_signup().await.unwrap());

        service
            .db
            .update_security_setting(Some("abc.com".to_owned()), false)
            .await
            .unwrap();

        assert!(service.allow_self_signup().await.unwrap());
    }

    #[tokio::test]
    async fn test_create_invitation_without_license() {
        let service = test_authentication_service_without_valid_license().await;
        assert_matches!(
            service.create_invitation("abc.com".into()).await,
            Err(CoreError::InvalidLicense(_))
        )
    }

    #[tokio::test]
    async fn test_create_invitation_without_sufficient_seats() {
        let service = test_authentication_service_with_license(Arc::new(
            MockLicenseService::team_with_seats(2),
        ))
        .await;
        assert_matches!(service.create_invitation("abc.com".into()).await, Ok(_));

        let service = test_authentication_service_with_license(Arc::new(
            MockLicenseService::team_with_seats(1),
        ))
        .await;
        assert_matches!(
            service.create_invitation("abc.com".into()).await,
            Err(CoreError::InvalidLicense(_))
        )
    }

    #[tokio::test]
    async fn test_update_user_active_on_admin_seats() {
        let service = test_authentication_service_with_license(Arc::new(
            MockLicenseService::team_with_seats(3),
        ))
        .await;

        // Create owner user.
        service
            .register("a@example.com".into(), "pass".into(), None, None)
            .await
            .unwrap();

        let user1 = service
            .db
            .create_user("b@example.com".into(), Some("pass".into()), false, None)
            .await
            .unwrap();
        let user2 = service
            .db
            .create_user("c@example.com".into(), Some("pass".into()), false, None)
            .await
            .unwrap();
        let user3 = service
            .db
            .create_user("d@example.com".into(), Some("pass".into()), false, None)
            .await
            .unwrap();

        service
            .update_user_role(&user1.as_id(), true)
            .await
            .unwrap();
        service
            .update_user_role(&user2.as_id(), true)
            .await
            .unwrap();

        assert_matches!(service.db.count_active_admin_users().await, Ok(3));

        assert_matches!(
            service.update_user_role(&user3.as_id(), true).await,
            Err(CoreError::InvalidLicense(_))
        );

        // Change user2 to deactive.
        service
            .update_user_active(&user2.as_id(), false)
            .await
            .unwrap();

        assert_matches!(service.db.count_active_admin_users().await, Ok(2));
        assert_matches!(service.update_user_role(&user3.as_id(), true).await, Ok(_));

        // Not able to toggle user to active due to admin seat limits.
        assert_matches!(
            service.update_user_role(&user2.as_id(), true).await,
            Err(CoreError::InvalidLicense(_))
        );
    }

    #[tokio::test]
    async fn test_update_password() {
        let service = test_authentication_service().await;
        let id = service
            .db
            .create_user(
                "test@example.com".into(),
                password_hash("pass").ok(),
                true,
                None,
            )
            .await
            .unwrap();

        let id = id.as_id();

        assert!(service
            .update_user_password(&id, Some("pass"), "newpass")
            .await
            .is_ok());

        assert!(service
            .update_user_password(&id, Some("wrong"), "newpass2")
            .await
            .is_err());

        assert!(service
            .update_user_password(&id, Some("newpass"), "newpass2")
            .await
            .is_ok());
    }

    #[tokio::test]
    async fn test_sso_user_forbid_update_password() {
        let service = test_authentication_service().await;
        let id = service
            .db
            .create_user("test@example.com".into(), None, true, None)
            .await
            .unwrap();

        let id = id.as_id();

        assert!(service
            .update_user_password(&id, None, "newpass2")
            .await
            .is_err());
    }

    #[tokio::test]
    async fn test_sso_user_forbid_update_name() {
        let service = test_authentication_service().await;
        let id = service
            .db
            .create_user("test@example.com".into(), None, true, None)
            .await
            .unwrap();

        assert!(service
            .update_user_name(&id.as_id(), "newname".into())
            .await
            .is_err());
    }

    #[tokio::test]
    async fn test_sso_user_forbid_generate_password_reset_url() {
        let service = test_authentication_service().await;
        let id = service
            .db
            .create_user("test@example.com".into(), None, true, None)
            .await
            .unwrap();

        assert!(service
            .generate_reset_password_url(&id.as_id())
            .await
            .is_err());
    }

    #[tokio::test]
    async fn test_sso_user_forbid_request_password_reset_email() {
        let service = test_authentication_service().await;
        let _id = service
            .db
            .create_user("test@example.com".into(), None, true, None)
            .await
            .unwrap();

        assert!(service
            .request_password_reset_email("test@example.com".into())
            .await
            .is_err());
    }

    #[tokio::test]
    async fn test_cannot_reset_same_password() {
        let (service, _mail) = test_authentication_service_with_mail().await;
        let id = service
            .db
            .create_user(
                "test@example.com".into(),
                password_hash("pass").ok(),
                true,
                None,
            )
            .await
            .unwrap();

        assert!(service
            .update_user_password(&id.as_id(), Some("pass"), "pass")
            .await
            .is_err());

        service
            .request_password_reset_email("test@example.com".into())
            .await
            .unwrap();
        let reset = service
            .db
            .get_password_reset_by_user_id(id as i64)
            .await
            .unwrap()
            .unwrap();
        assert!(service.password_reset(&reset.code, "pass").await.is_err());
    }

    #[tokio::test]
    async fn test_logout_all_sessions() {
        let service = test_authentication_service().await;

        let id = service
            .db
            .create_user(
                "test@example.com".into(),
                password_hash("pass").ok(),
                true,
                None,
            )
            .await
            .unwrap();

        let token = service
            .token_auth("test@example.com".into(), "pass".into())
            .await
            .unwrap();

        service.logout_all_sessions(&id.as_id()).await.unwrap();

        assert!(service.refresh_token(token.refresh_token).await.is_err());
    }

    #[tokio::test]
    async fn test_ldap_credential() {
        let service = test_authentication_service().await;
        service
            .update_ldap_credential(UpdateLdapCredentialInput {
                host: "ldap.example.com".into(),
                port: 389,
                bind_dn: "cn=admin,dc=example,dc=com".into(),
                bind_password: Some("password".into()),
                base_dn: "dc=example,dc=com".into(),
                user_filter: "(&(objectClass=person)(uid=%s))".into(),
                encryption: LdapEncryptionKind::None,
                skip_tls_verify: false,
                email_attribute: "mail".into(),
                name_attribute: Some("cn".into()),
            })
            .await
            .unwrap();

        // test the read_ldap_credential
        let cred = service.read_ldap_credential().await.unwrap().unwrap();
        assert_eq!(cred.host, "ldap.example.com");
        assert_eq!(cred.port, 389);
        assert_eq!(cred.bind_dn, "cn=admin,dc=example,dc=com");
        assert_eq!(cred.base_dn, "dc=example,dc=com");
        assert_eq!(cred.user_filter, "(&(objectClass=person)(uid=%s))");
        assert_eq!(cred.encryption, LdapEncryptionKind::None);
        assert!(!cred.skip_tls_verify);
        assert_eq!(cred.email_attribute, "mail");
        assert_eq!(cred.name_attribute, Some("cn".into()));

        service
            .update_ldap_credential(UpdateLdapCredentialInput {
                host: "ldap1.example1.com".into(),
                port: 3890,
                bind_dn: "cn=admin1,dc=example1,dc=com".into(),
                bind_password: None,
                base_dn: "dc=example1,dc=com".into(),
                user_filter: "((uid=%s))".into(),
                encryption: LdapEncryptionKind::None,
                skip_tls_verify: true,
                email_attribute: "email".into(),
                name_attribute: Some("name".into()),
            })
            .await
            .unwrap();

        // use db to verify the update and password sine it's not returned in service
        let cred = service.db.read_ldap_credential().await.unwrap().unwrap();
        assert_eq!(cred.host, "ldap1.example1.com");
        assert_eq!(cred.port, 3890);
        assert_eq!(cred.bind_dn, "cn=admin1,dc=example1,dc=com");
        assert_eq!(cred.bind_password, "password");
        assert_eq!(cred.base_dn, "dc=example1,dc=com");
        assert_eq!(cred.user_filter, "((uid=%s))");
        assert_eq!(cred.encryption, "none");
        assert!(cred.skip_tls_verify);
        assert_eq!(cred.email_attribute, "email");
        assert_eq!(cred.name_attribute, Some("name".into()));
    }

    #[tokio::test]
    async fn test_oauth_credential() {
        let service = test_authentication_service().await;
        service
            .update_oauth_credential(UpdateOAuthCredentialInput {
                provider: OAuthProvider::Google,
                client_id: "id".into(),
                client_secret: Some("secret".into()),
            })
            .await
            .unwrap();

        let cred = service
            .read_oauth_credential(OAuthProvider::Google)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(cred.provider, OAuthProvider::Google);
        assert_eq!(cred.client_id, "id");
        assert_eq!(cred.client_secret, "secret");
    }

    #[tokio::test]
    async fn test_ldap_login() {
        let service = test_authentication_service().await;
        let license = LicenseInfo {
            r#type: LicenseType::Enterprise,
            status: LicenseStatus::Ok,
            seats: 1000,
            seats_used: 0,
            issued_at: None,
            expires_at: None,
        };

        service
            .create_invitation("user@example.com".into())
            .await
            .unwrap();
        let mut ldap_client = FakeLdapClient { state: "" };

        let response = ldap_login(
            &mut ldap_client,
            &service.db,
            &*service.setting,
            &license,
            &*service.mail,
            "user",
            "password",
        )
        .await
        .unwrap();

        assert!(!response.refresh_token.is_empty());
    }

    #[tokio::test]
    async fn test_ldap_login_not_found() {
        let service = test_authentication_service().await;
        let license = LicenseInfo {
            r#type: LicenseType::Enterprise,
            status: LicenseStatus::Ok,
            seats: 1000,
            seats_used: 0,
            issued_at: None,
            expires_at: None,
        };

        service
            .create_invitation("user@example.com".into())
            .await
            .unwrap();
        let mut ldap_client = FakeLdapClient { state: "not_found" };

        let response = ldap_login(
            &mut ldap_client,
            &service.db,
            &*service.setting,
            &license,
            &*service.mail,
            "user",
            "password",
        )
        .await;

        assert!(response.is_err());
    }

    #[tokio::test]
    async fn test_oauth_login() {
        let service = test_authentication_service().await;
        let license = LicenseInfo {
            r#type: LicenseType::Enterprise,
            status: LicenseStatus::Ok,
            seats: 1000,
            seats_used: 0,
            issued_at: None,
            expires_at: None,
        };

        let client = Arc::new(TestOAuthClient {
            access_token_response: || Ok("faketoken".into()),
            user_email: "user@example.com".into(),
            user_name: "user".into(),
        });

        service
            .create_invitation("user@example.com".into())
            .await
            .unwrap();

        let response = oauth_login(
            client,
            "fakecode".into(),
            &service.db,
            &*service.setting,
            &license,
            &*service.mail,
        )
        .await
        .unwrap();

        assert!(!response.access_token.is_empty());

        let client = Arc::new(TestOAuthClient {
            access_token_response: || Err(anyhow!("bad auth")),
            user_email: "user@example.com".into(),
            user_name: "user".into(),
        });

        let response = oauth_login(
            client,
            "fakecode".into(),
            &service.db,
            &*service.setting,
            &license,
            &*service.mail,
        )
        .await;

        assert!(response.is_err());
    }

    #[tokio::test]
    async fn test_personate_user_auth_token() {
        std::env::set_var("TABBY_OWNER_IMPERSONATE_OVERRIDE", "abc@example.com:123456");
        let service = test_authentication_service().await;
        register_admin_user(&service).await;
        assert!(service
            .token_auth("abc@example.com".to_owned(), "123456".to_owned())
            .await
            .is_ok());
    }

    #[tokio::test]
    #[serial]
    async fn test_generate_reset_password_url() {
        let (service, _smtp) = test_authentication_service_with_mail().await;

        // Create an active user
        let _id = service
            .db
            .create_user(
                "active_user@example.com".into(),
                Some("pass".into()),
                false,
                None,
            )
            .await
            .unwrap();
        let active_user = service
            .get_user_by_email("active_user@example.com")
            .await
            .unwrap();

        // Test generating reset URL for an active user
        let url = service
            .generate_reset_password_url(&active_user.id)
            .await
            .unwrap();
        assert!(url.contains("/auth/reset-password?code="));

        // Create an inactive user
        let id = service
            .db
            .create_user(
                "inactive_user@example.com".into(),
                Some("pass".into()),
                false,
                None,
            )
            .await
            .unwrap();
        service
            .update_user_active(&id.as_id(), false)
            .await
            .unwrap();
        let inactive_user = service
            .get_user_by_email("inactive_user@example.com")
            .await
            .unwrap();

        // Test generating reset URL for an inactive user
        let result = service.generate_reset_password_url(&inactive_user.id).await;
        assert!(result.is_err());
    }
}
