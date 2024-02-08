use hash_ids::HashIds;
use lazy_static::lazy_static;
use tabby_db::{
    EmailSettingDAO, GithubOAuthCredentialDAO, GoogleOAuthCredentialDAO, InvitationDAO, JobRunDAO,
    RepositoryDAO, UserDAO,
};

use crate::schema::{
    auth::{self, OAuthCredential, OAuthProvider},
    email::EmailSetting,
    job,
    repository::Repository,
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
            job: run.job,
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

impl From<EmailSettingDAO> for EmailSetting {
    fn from(value: EmailSettingDAO) -> Self {
        EmailSetting {
            smtp_username: value.smtp_username,
            smtp_server: value.smtp_server,
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
