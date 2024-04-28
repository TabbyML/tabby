use std::{pin::Pin, sync::Arc};

use futures::Future;
use juniper::ID;
use tokio_cron_scheduler::{Job, JobScheduler};
use tracing::{debug, warn};

use crate::schema::job::JobService;

pub struct JobController {
    scheduler: JobScheduler,
    service: Arc<dyn JobService>,
}

impl JobController {
    pub async fn new(service: Arc<dyn JobService>) -> Self {
        service.cleanup().await.expect("failed to cleanup jobs");
        let scheduler = JobScheduler::new()
            .await
            .expect("failed to create job scheduler");
        Self { scheduler, service }
    }

    pub async fn run(&self) {
        self.scheduler
            .start()
            .await
            .expect("failed to start job scheduler")
    }

    /// Register a new job with the scheduler, the job will be displayed in Jobs dashboard.
    pub async fn register_public<T>(&mut self, name: &str, schedule: &str, func: T)
    where
        T: FnMut(&JobContext) -> Pin<Box<dyn Future<Output = anyhow::Result<i32>> + Send>>
            + Send
            + Sync
            + Clone
            + 'static,
    {
        let job_mutex = Arc::new(tokio::sync::Mutex::new(()));
        let service = self.service.clone();
        let name = name.to_owned();
        let func = func.clone();
        let job = Job::new_async(schedule, move |uuid, mut scheduler| {
            let job_mutex = job_mutex.clone();
            let service = service.clone();
            let name = name.clone();
            let mut func = func.clone();
            Box::pin(async move {
                let Ok(_guard) = job_mutex.try_lock() else {
                    warn!("Scheduler job overlapped, skipping...");
                    return;
                };

                debug!("Running public job `{}`", name);

                let Ok(id) = service.start(name.clone()).await else {
                    warn!("failed to create job `{}`", &name);
                    return;
                };

                let context = JobContext::new(id.clone(), service.clone());
                match func(&context).await {
                    Ok(exit_code) => {
                        debug!("Job `{}` completed with exit code {}", &name, exit_code);
                        let _ = service.complete(&id, exit_code).await;
                    }
                    Err(e) => {
                        warn!("Job `{}` failed: {}", &name, e);
                        let _ = service.complete(&id, -1).await;
                    }
                }

                if let Ok(Some(next_tick)) = scheduler.next_tick_for_job(uuid).await {
                    debug!(
                        "Next time for job `{}` is {:?}",
                        &name,
                        next_tick.with_timezone(&chrono::Local)
                    );
                }
            })
        })
        .expect("failed to create job");

        self.scheduler.add(job).await.expect("failed to add job");
    }

    /// Register a new job with the scheduler, the job will NOT be displayed in Jobs dashboard.
    pub async fn register<T>(&mut self, name: &str, schedule: &str, func: T)
    where
        T: FnMut() -> Pin<Box<dyn Future<Output = anyhow::Result<()>> + Send>>
            + Send
            + Sync
            + Clone
            + 'static,
    {
        let job_mutex = Arc::new(tokio::sync::Mutex::new(()));
        let func = func.clone();
        let name = name.to_owned();
        let job = Job::new_async(schedule, move |uuid, mut scheduler| {
            let job_mutex = job_mutex.clone();
            let name = name.clone();
            let mut func = func.clone();
            Box::pin(async move {
                let Ok(_guard) = job_mutex.try_lock() else {
                    warn!("Scheduler job overlapped, skipping...");
                    return;
                };

                debug!("Running job `{}`", name);
                match func().await {
                    Ok(_) => {
                        debug!("Job `{}` completed", name);
                    }
                    Err(e) => {
                        warn!("Job `{}` failed: {}", name, e);
                    }
                }

                if let Ok(Some(next_tick)) = scheduler.next_tick_for_job(uuid).await {
                    debug!(
                        "Next time for job `{}` is {:?}",
                        &name,
                        next_tick.with_timezone(&chrono::Local)
                    );
                }
            })
        })
        .expect("failed to create job");

        self.scheduler.add(job).await.expect("failed to add job");
    }
}

#[derive(Clone)]
pub struct JobContext {
    id: ID,
    service: Arc<dyn JobService>,
}

impl JobContext {
    pub fn new(id: ID, service: Arc<dyn JobService>) -> Self {
        Self { id, service }
    }

    pub async fn stdout_writeline(&self, stdout: String) {
        let stdout = stdout + "\n";
        match self.service.update_stdout(&self.id, stdout).await {
            Ok(_) => (),
            Err(_) => {
                warn!("Failed to write stdout to job `{}`", self.id);
            }
        }
    }

    pub async fn stderr_writeline(&self, stderr: String) {
        let stderr = stderr + "\n";
        match self.service.update_stderr(&self.id, stderr).await {
            Ok(_) => (),
            Err(_) => {
                warn!("Failed to write stderr to job `{}`", self.id);
            }
        }
    }
}
