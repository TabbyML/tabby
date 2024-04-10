use anyhow::anyhow;
use hash_ids::HashIds;
use lazy_static::lazy_static;
use tabby_db::{
    EmailSettingDAO, GithubProvidedRepositoryDAO, GithubRepositoryProviderDAO, InvitationDAO,
    JobRunDAO, OAuthCredentialDAO, RepositoryDAO, ServerSettingDAO, UserDAO,
};

use crate::schema::{
    auth::{self, OAuthCredential, OAuthProvider},
    email::{AuthMethod, EmailSetting, Encryption},
    github_repository_provider::{GithubProvidedRepository, GithubRepositoryProvider},
    job,
    repository::Repository,
    setting::{NetworkSetting, SecuritySetting},
    CoreError,
};

impl From<InvitationDAO> for auth::Invitation {
    fn from(val: InvitationDAO) -> Self {
        Self {
            id: (val.id as i32).as_id(),
            email: val.email,
            code: val.code,
            created_at: *val.created_at,
        }
    }
}

impl From<JobRunDAO> for job::JobRun {
    fn from(run: JobRunDAO) -> Self {
        Self {
            id: run.id.as_id(),
            job: run.name,
            created_at: *run.created_at,
            updated_at: *run.updated_at,
            finished_at: run.finished_at.into_option(),
            exit_code: run.exit_code.map(|i| i as i32),
            stdout: run.stdout,
            stderr: run.stderr,
        }
    }
}

impl From<UserDAO> for auth::User {
    fn from(val: UserDAO) -> Self {
        let is_owner = val.is_owner();
        auth::User {
            id: val.id.as_id(),
            email: val.email,
            is_owner,
            is_admin: val.is_admin,
            auth_token: val.auth_token,
            created_at: *val.created_at,
            active: val.active,
            is_password_set: val.password_encrypted.is_some(),
        }
    }
}

impl TryFrom<OAuthCredentialDAO> for OAuthCredential {
    type Error = anyhow::Error;

    fn try_from(val: OAuthCredentialDAO) -> Result<Self, Self::Error> {
        Ok(OAuthCredential {
            provider: OAuthProvider::from_enum_str(&val.provider)?,
            client_id: val.client_id,
            created_at: *val.created_at,
            updated_at: *val.updated_at,
            client_secret: Some(val.client_secret),
        })
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
        let encryption = Encryption::from_enum_str(&value.encryption)?;
        let auth_method = AuthMethod::from_enum_str(&value.auth_method)?;

        Ok(EmailSetting {
            smtp_username: value.smtp_username,
            smtp_server: value.smtp_server,
            smtp_port: value.smtp_port as i32,
            from_address: value.from_address,
            encryption,
            auth_method,
        })
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

impl From<GithubRepositoryProviderDAO> for GithubRepositoryProvider {
    fn from(value: GithubRepositoryProviderDAO) -> Self {
        Self {
            display_name: value.display_name,
            application_id: value.application_id,
            id: value.id.as_id(),
        }
    }
}

impl From<GithubProvidedRepositoryDAO> for GithubProvidedRepository {
    fn from(value: GithubProvidedRepositoryDAO) -> Self {
        Self {
            id: value.id.as_id(),
            github_repository_provider_id: value.github_repository_provider_id.as_id(),
            name: value.name,
            git_url: value.git_url,
            vendor_id: value.vendor_id,
            active: value.active,
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
    fn as_rowid(&self) -> std::result::Result<i64, CoreError>;
}

impl AsRowid for juniper::ID {
    fn as_rowid(&self) -> std::result::Result<i64, CoreError> {
        HASHER
            .decode(self)
            .first()
            .map(|i| *i as i64)
            .ok_or(CoreError::InvalidID)
    }
}

pub trait AsID {
    fn as_id(&self) -> juniper::ID;
}

impl AsID for i64 {
    fn as_id(&self) -> juniper::ID {
        juniper::ID::new(HASHER.encode(&[*self as u64]))
    }
}

impl AsID for i32 {
    fn as_id(&self) -> juniper::ID {
        (*self as i64).as_id()
    }
}

pub trait DbEnum: Sized {
    fn as_enum_str(&self) -> &'static str;
    fn from_enum_str(s: &str) -> anyhow::Result<Self>;
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

impl DbEnum for OAuthProvider {
    fn as_enum_str(&self) -> &'static str {
        match self {
            OAuthProvider::Google => "google",
            OAuthProvider::Github => "github",
        }
    }

    fn from_enum_str(s: &str) -> anyhow::Result<Self> {
        match s {
            "github" => Ok(OAuthProvider::Github),
            "google" => Ok(OAuthProvider::Google),
            _ => Err(anyhow!("Invalid OAuth credential type")),
        }
    }
}

impl DbEnum for AuthMethod {
    fn as_enum_str(&self) -> &'static str {
        match self {
            AuthMethod::None => "none",
            AuthMethod::Plain => "plain",
            AuthMethod::Login => "login",
        }
    }

    fn from_enum_str(s: &str) -> anyhow::Result<Self> {
        match s {
            "none" => Ok(AuthMethod::None),
            "plain" => Ok(AuthMethod::Plain),
            "login" => Ok(AuthMethod::Login),
            _ => Err(anyhow!("{s} is not a valid value for AuthMethod")),
        }
    }
}
