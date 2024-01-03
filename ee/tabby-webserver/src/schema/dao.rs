use tabby_db::{GithubOAuthCredentialDAO, InvitationDAO, JobRunDAO, UserDAO};

use crate::schema::{
    auth,
    job,
};

impl From<InvitationDAO> for auth::InvitationNext {
    fn from(val: InvitationDAO) -> Self {
        Self {
            id: juniper::ID::new(val.id.to_string()),
            email: val.email,
            code: val.code,
            created_at: val.created_at,
        }
    }
}

impl From<JobRunDAO> for job::JobRun {
    fn from(run: JobRunDAO) -> Self {
        Self {
            id: juniper::ID::new(run.id.to_string()),
            job_name: run.job_name,
            start_time: run.start_time,
            finish_time: run.finish_time,
            exit_code: run.exit_code,
            stdout: run.stdout,
            stderr: run.stderr,
        }
    }
}

impl From<UserDAO> for auth::User {
    fn from(val: UserDAO) -> Self {
        auth::User {
            id: juniper::ID::new(val.id.to_string()),
            email: val.email,
            is_admin: val.is_admin,
            auth_token: val.auth_token,
            created_at: val.created_at,
        }
    }
}

impl From<GithubOAuthCredentialDAO> for auth::OAuthCredential {
    fn from(val: GithubOAuthCredentialDAO) -> Self {
        auth::OAuthCredential {
            provider: auth::OAuthProvider::Github,
            client_id: val.client_id,
            client_secret: val.client_secret,
            active: val.active,
            created_at: val.created_at,
            updated_at: val.updated_at,
        }
    }
}
