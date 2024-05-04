use std::{
    fmt::Debug,
    task::{Context, Poll},
};

use apalis::prelude::Request;
use futures::{future::BoxFuture, FutureExt};
use tabby_db::DbConn;
use tower::{Layer, Service};
use tracing::{debug, warn};

#[derive(Clone)]
pub struct JobLogger {
    id: i64,
    db: DbConn,
}

impl JobLogger {
    async fn new(name: &'static str, db: DbConn) -> Self {
        let id = db
            .create_job_run(name.to_owned())
            .await
            .expect("failed to create job");
        Self { id, db }
    }

    pub async fn r#internal_println(&self, stdout: String) {
        let stdout = stdout + "\n";
        match self.db.update_job_stdout(self.id, stdout).await {
            Ok(_) => (),
            Err(_) => {
                warn!("Failed to write stdout to job `{}`", self.id);
            }
        }
    }

    pub async fn r#internal_eprintln(&self, stderr: String) {
        let stderr = stderr + "\n";
        match self.db.update_job_stderr(self.id, stderr).await {
            Ok(_) => (),
            Err(_) => {
                warn!("Failed to write stderr to job `{}`", self.id);
            }
        }
    }

    async fn complete(&mut self, exit_code: i32) {
        match self.db.update_job_status(self.id, exit_code).await {
            Ok(_) => (),
            Err(_) => {
                warn!("Failed to complete job `{}`", self.id);
            }
        }
    }
}

pub struct JobLogLayer {
    db: DbConn,
    name: &'static str,
}

impl JobLogLayer {
    pub fn new(db: DbConn, name: &'static str) -> Self {
        Self { db, name }
    }
}

impl<S> Layer<S> for JobLogLayer {
    type Service = JobLogService<S>;

    fn layer(&self, service: S) -> Self::Service {
        JobLogService {
            db: self.db.clone(),
            name: self.name,
            service,
        }
    }
}

#[derive(Clone)]
pub struct JobLogService<S> {
    db: DbConn,
    name: &'static str,
    service: S,
}

impl<S, Req> Service<Request<Req>> for JobLogService<S>
where
    S: Service<Request<Req>> + Clone,
    Request<Req>: Send + 'static,
    S: Send + 'static,
    S::Future: Send + 'static,
    S::Response: Send + 'static,
    S::Error: Send + Debug + 'static,
{
    type Response = ();
    type Error = S::Error;
    type Future = BoxFuture<'static, Result<Self::Response, Self::Error>>;

    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        self.service.poll_ready(cx)
    }

    fn call(&mut self, mut request: Request<Req>) -> Self::Future {
        debug!("Starting job `{}`", self.name);
        let name = self.name;
        let db = self.db.clone();
        let mut service = self.service.clone();
        let fut_with_log = async move {
            let mut logger = JobLogger::new(name, db).await;
            request.insert(logger.clone());
            match service.call(request).await {
                Ok(_) => {
                    debug!("Job `{}` completed", name);
                    logger.complete(0).await;
                    Ok(())
                }
                Err(e) => {
                    warn!("Job `{}` failed: {:?}", name, e);
                    logger.complete(-1).await;
                    Err(e)
                }
            }
        };
        fut_with_log.boxed()
    }
}
