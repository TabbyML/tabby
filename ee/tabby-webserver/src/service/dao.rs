use anyhow::anyhow;
use hash_ids::HashIds;
use lazy_static::lazy_static;
use tabby_db::{
    DbEnum, EmailSettingDAO, GithubOAuthCredentialDAO, GoogleOAuthCredentialDAO, InvitationDAO,
    JobRunDAO, RepositoryDAO, ServerSettingDAO, UserDAO,
};
use validator::validate_url;

use crate::schema::{
    auth::{self, OAuthCredential, OAuthProvider},
    email::{AuthMethod, EmailSetting, Encryption},
    job,
    repository::Repository,
    setting::{NetworkSetting, SecuritySetting},
    CoreError,
};

impl From<InvitationDAO> for auth::Invitation {
    fn from(val: InvitationDAO) -> Self {
        Self {
            id: val.id.as_id(),
            email: val.email,
            code: val.code,
            created_at: val.created_at,
        }
    }
}

impl From<JobRunDAO> for job::JobRun {
    fn from(run: JobRunDAO) -> Self {
        Self {
            id: run.id.as_id(),
            job: run.name,
            created_at: run.created_at,
            updated_at: run.updated_at,
            finished_at: run.finished_at,
            exit_code: run.exit_code,
            stdout: run.stdout,
            stderr: run.stderr,
        }
    }
}

impl From<UserDAO> for auth::User {
    fn from(val: UserDAO) -> Self {
        auth::User {
            id: val.id.as_id(),
            email: val.email,
            is_admin: val.is_admin,
            auth_token: val.auth_token,
            created_at: val.created_at,
            active: val.active,
        }
    }
}

impl From<GithubOAuthCredentialDAO> for OAuthCredential {
    fn from(val: GithubOAuthCredentialDAO) -> Self {
        OAuthCredential {
            provider: OAuthProvider::Github,
            client_id: val.client_id,
            client_secret: val.client_secret,
            redirect_uri: None,
            created_at: val.created_at,
            updated_at: val.updated_at,
        }
    }
}

impl From<GoogleOAuthCredentialDAO> for OAuthCredential {
    fn from(val: GoogleOAuthCredentialDAO) -> Self {
        OAuthCredential {
            provider: OAuthProvider::Google,
            client_id: val.client_id,
            client_secret: val.client_secret,
            redirect_uri: Some(val.redirect_uri),
            created_at: val.created_at,
            updated_at: val.updated_at,
        }
    }
}

impl From<RepositoryDAO> for Repository {
    fn from(value: RepositoryDAO) -> Self {
        Repository {
            id: value.id.as_id(),
            name: value.name,
            git_url: value.git_url,
        }
    }
}

impl TryFrom<EmailSettingDAO> for EmailSetting {
    type Error = anyhow::Error;

    fn try_from(value: EmailSettingDAO) -> Result<Self, Self::Error> {
        EmailSetting::new_validate(
            value.smtp_username,
            value.smtp_server,
            value.from_address,
            value.encryption,
            value.auth_method,
        )
    }
}

impl From<ServerSettingDAO> for SecuritySetting {
    fn from(value: ServerSettingDAO) -> Self {
        Self {
            allowed_register_domain_list: value
                .security_allowed_register_domain_list()
                .map(|s| s.to_owned())
                .collect(),
            disable_client_side_telemetry: value.security_disable_client_side_telemetry,
        }
    }
}

impl From<ServerSettingDAO> for NetworkSetting {
    fn from(value: ServerSettingDAO) -> Self {
        Self {
            external_url: value.network_external_url,
        }
    }
}

lazy_static! {
    static ref HASHER: HashIds = HashIds::builder()
        .with_salt("tabby-id-serializer")
        .with_min_length(6)
        .finish();
}

pub trait AsRowid {
    fn as_rowid(&self) -> std::result::Result<i32, CoreError>;
}

impl AsRowid for juniper::ID {
    fn as_rowid(&self) -> std::result::Result<i32, CoreError> {
        HASHER
            .decode(self)
            .first()
            .map(|i| *i as i32)
            .ok_or(CoreError::InvalidIDError)
    }
}

pub trait AsID {
    fn as_id(&self) -> juniper::ID;
}

impl AsID for i32 {
    fn as_id(&self) -> juniper::ID {
        juniper::ID::new(HASHER.encode(&[*self as u64]))
    }
}

impl DbEnum for Encryption {
    fn as_enum_str(&self) -> &'static str {
        match self {
            Encryption::StartTls => "starttls",
            Encryption::SslTls => "ssltls",
            Encryption::None => "none",
        }
    }

    fn from_enum_str(s: &str) -> anyhow::Result<Self> {
        match s {
            "starttls" => Ok(Encryption::StartTls),
            "ssltls" => Ok(Encryption::SslTls),
            "none" => Ok(Encryption::None),
            _ => Err(anyhow!("{s} is not a valid value for Encryption")),
        }
    }
}

impl DbEnum for AuthMethod {
    fn as_enum_str(&self) -> &'static str {
        match self {
            AuthMethod::Plain => "plain",
            AuthMethod::Login => "login",
            AuthMethod::XOAuth2 => "xoauth2",
        }
    }

    fn from_enum_str(s: &str) -> anyhow::Result<Self> {
        match s {
            "plain" => Ok(AuthMethod::Plain),
            "login" => Ok(AuthMethod::Login),
            "xoauth2" => Ok(AuthMethod::XOAuth2),
            _ => Err(anyhow!("{s} is not a valid value for AuthMethod")),
        }
    }
}

impl EmailSetting {
    pub fn new_validate(
        smtp_username: String,
        smtp_server: String,
        from_address: String,
        encryption: String,
        auth_method: String,
    ) -> anyhow::Result<Self> {
        if !validate_url(&smtp_server) {
            return Err(anyhow!("Invalid smtp server address"));
        }

        let encryption = Encryption::from_enum_str(&encryption)?;
        let auth_method = AuthMethod::from_enum_str(&auth_method)?;

        Ok(EmailSetting {
            smtp_username,
            smtp_server,
            from_address,
            encryption,
            auth_method,
        })
    }
}
