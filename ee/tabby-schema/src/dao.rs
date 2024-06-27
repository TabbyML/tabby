use anyhow::bail;
use hash_ids::HashIds;
use lazy_static::lazy_static;
use tabby_db::{
    EmailSettingDAO, IntegrationDAO, InvitationDAO, JobRunDAO, OAuthCredentialDAO,
    ServerSettingDAO, UserDAO, UserEventDAO,
};

use crate::{
    integration::{Integration, IntegrationKind, IntegrationStatus},
    repository::{ProvidedRepository, RepositoryKind},
    schema::{
        auth::{self, OAuthCredential, OAuthProvider},
        email::{AuthMethod, EmailSetting, Encryption},
        job,
        repository::{
            GithubProvidedRepository, GithubRepositoryProvider, GitlabProvidedRepository,
            GitlabRepositoryProvider, RepositoryProviderStatus,
        },
        setting::{NetworkSetting, SecuritySetting},
        user_event::{EventKind, UserEvent},
        CoreError,
    },
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
            started_at: run.started_at.into_option(),
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
            name: val.name.unwrap_or_default(),
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
            client_secret: val.client_secret,
        })
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

impl TryFrom<IntegrationDAO> for Integration {
    type Error = anyhow::Error;
    fn try_from(value: IntegrationDAO) -> anyhow::Result<Self> {
        let status = if value.synced && value.error.is_none() {
            IntegrationStatus::Ready
        } else if value.error.is_some() {
            IntegrationStatus::Failed
        } else {
            IntegrationStatus::Pending
        };
        Ok(Self {
            id: value.id.as_id(),
            kind: IntegrationKind::from_enum_str(&value.kind)?,
            display_name: value.display_name,
            access_token: value.access_token,
            api_base: value.api_base,
            created_at: *value.created_at,
            updated_at: *value.updated_at,
            status,
        })
    }
}

impl From<IntegrationKind> for RepositoryKind {
    fn from(value: IntegrationKind) -> Self {
        match value {
            IntegrationKind::Github => RepositoryKind::Github,
            IntegrationKind::Gitlab => RepositoryKind::Gitlab,
            IntegrationKind::GithubSelfHosted => RepositoryKind::GithubSelfHosted,
            IntegrationKind::GitlabSelfHosted => RepositoryKind::GitlabSelfHosted,
        }
    }
}

impl From<ProvidedRepository> for GithubProvidedRepository {
    fn from(value: ProvidedRepository) -> Self {
        Self {
            id: value.id,
            vendor_id: value.vendor_id,
            github_repository_provider_id: value.integration_id,
            name: value.display_name,
            git_url: value.git_url,
            active: value.active,
            refs: value.refs,
        }
    }
}

impl From<Integration> for GithubRepositoryProvider {
    fn from(value: Integration) -> Self {
        Self {
            id: value.id,
            display_name: value.display_name,
            status: value.status.into(),
            access_token: Some(value.access_token),
            api_base: value.api_base,
        }
    }
}

impl From<Integration> for GitlabRepositoryProvider {
    fn from(value: Integration) -> Self {
        Self {
            id: value.id,
            display_name: value.display_name,
            status: value.status.into(),
            access_token: Some(value.access_token),
            api_base: value.api_base,
        }
    }
}

impl From<IntegrationStatus> for RepositoryProviderStatus {
    fn from(value: IntegrationStatus) -> Self {
        match value {
            IntegrationStatus::Ready => Self::Ready,
            IntegrationStatus::Pending => Self::Pending,
            IntegrationStatus::Failed => Self::Failed,
        }
    }
}

impl From<ProvidedRepository> for GitlabProvidedRepository {
    fn from(value: ProvidedRepository) -> Self {
        Self {
            id: value.id,
            vendor_id: value.vendor_id,
            gitlab_repository_provider_id: value.integration_id,
            name: value.display_name,
            git_url: value.git_url,
            active: value.active,
            refs: value.refs,
        }
    }
}

impl TryFrom<UserEventDAO> for UserEvent {
    type Error = anyhow::Error;
    fn try_from(value: UserEventDAO) -> Result<Self, Self::Error> {
        Ok(Self {
            id: value.id.as_id(),
            user_id: value.user_id.as_id(),
            kind: EventKind::from_enum_str(&value.kind)?,
            created_at: value.created_at.into(),
            payload: String::from_utf8(value.payload)?,
        })
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

impl DbEnum for EventKind {
    fn as_enum_str(&self) -> &'static str {
        match self {
            EventKind::Completion => "completion",
            EventKind::Select => "select",
            EventKind::View => "view",
            EventKind::Dismiss => "dismiss",
        }
    }

    fn from_enum_str(s: &str) -> anyhow::Result<Self> {
        match s {
            "completion" => Ok(EventKind::Completion),
            "select" => Ok(EventKind::Select),
            "view" => Ok(EventKind::View),
            "dismiss" => Ok(EventKind::Dismiss),
            _ => bail!("{s} is not a valid value for EventKind"),
        }
    }
}

impl DbEnum for IntegrationKind {
    fn as_enum_str(&self) -> &'static str {
        match self {
            IntegrationKind::Github => "github",
            IntegrationKind::Gitlab => "gitlab",
            IntegrationKind::GithubSelfHosted => "github_self_hosted",
            IntegrationKind::GitlabSelfHosted => "gitlab_self_hosted",
        }
    }

    fn from_enum_str(s: &str) -> anyhow::Result<Self> {
        match s {
            "github" => Ok(IntegrationKind::Github),
            "gitlab" => Ok(IntegrationKind::Gitlab),
            "github_self_hosted" => Ok(IntegrationKind::GithubSelfHosted),
            "gitlab_self_hosted" => Ok(IntegrationKind::GitlabSelfHosted),
            _ => bail!("{s} is not a valid value for ProviderKind"),
        }
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
            _ => bail!("{s} is not a valid value for Encryption"),
        }
    }
}

impl DbEnum for OAuthProvider {
    fn as_enum_str(&self) -> &'static str {
        match self {
            OAuthProvider::Google => "google",
            OAuthProvider::Github => "github",
            OAuthProvider::Gitlab => "gitlab",
        }
    }

    fn from_enum_str(s: &str) -> anyhow::Result<Self> {
        match s {
            "github" => Ok(OAuthProvider::Github),
            "google" => Ok(OAuthProvider::Google),
            "gitlab" => Ok(OAuthProvider::Gitlab),
            _ => bail!("Invalid OAuth credential type"),
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
            _ => bail!("{s} is not a valid value for AuthMethod"),
        }
    }
}
