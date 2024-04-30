use std::{collections::HashMap, pin::Pin, sync::Arc, time::Duration};

use futures::Future;
use juniper::ID;
use rand::Rng;
use tokio_cron_scheduler::{Job, JobScheduler};
use tracing::{debug, warn};

use crate::schema::job::JobService;

pub struct JobController {
    scheduler: JobScheduler,
    service: Arc<dyn JobService>,
    job_registry: HashMap<
        String,
        Arc<dyn Fn() -> Pin<Box<dyn Future<Output = ()> + Send>> + Send + Sync + 'static>,
    >,
}

impl JobController {
    pub async fn new(service: Arc<dyn JobService>) -> Self {
        service.cleanup().await.expect("failed to cleanup jobs");
        let scheduler = JobScheduler::new()
            .await
            .expect("failed to create job scheduler");
        Self {
            scheduler,
            service,
            job_registry: HashMap::default(),
        }
    }

    pub fn schedule(&self, name: &str) {
        let func = self
            .job_registry
            .get(name)
            .expect("failed to get job")
            .clone();
        let mut rng = rand::thread_rng();
        let delay = rng.gen_range(1..5);
        let _ = tokio::spawn(async move {
            tokio::time::sleep(Duration::from_secs(5 + delay)).await;
            func().await;
        });
    }

    fn run_oneshot(&self) {
        warn!(
            "Running controller job as oneshot, this should only be used for debugging purpose..."
        );
        for name in self.job_registry.keys() {
            self.schedule(name);
        }
    }

    pub async fn run(&self) {
        if std::env::var("TABBY_WEBSERVER_CONTROLLER_ONESHOT").is_ok() {
            self.run_oneshot();
        } else {
            self.scheduler
                .start()
                .await
                .expect("failed to start job scheduler")
        }
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
        self.register_impl(true, name, schedule, func).await;
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
        self.register_impl(false, name, schedule, move |_| {
            let mut func = func.clone();
            Box::pin(async move {
                func().await?;
                Ok(0)
            })
        })
        .await;
    }

    async fn register_impl<F>(&mut self, is_public: bool, name: &str, schedule: &str, func: F)
    where
        F: FnMut(&JobContext) -> Pin<Box<dyn Future<Output = anyhow::Result<i32>> + Send>>
            + Send
            + Sync
            + Clone
            + 'static,
    {
        let cloned_name = name.to_owned();
        let job_mutex = Arc::new(tokio::sync::Mutex::new(()));
        let service = self.service.clone();
        self.job_registry.insert(
            cloned_name.clone(),
            Arc::new(move || {
                let job_mutex = job_mutex.clone();
                let service = service.clone();
                let name = cloned_name.clone();
                let mut func = func.clone();

                Box::pin(async move {
                    let Ok(_guard) = job_mutex.try_lock() else {
                        warn!("Job `{}` overlapped, skipping...", name);
                        return;
                    };

                    debug!("Running job `{}`", name);
                    let context = JobContext::new(is_public, &name, service.clone()).await;
                    match func(&context).await {
                        Ok(exit_code) => {
                            debug!("Job `{}` completed with exit code {}", &name, exit_code);
                            context.complete(exit_code).await;
                        }
                        Err(e) => {
                            warn!("Job `{}` failed: {}", &name, e);
                            context.complete(-1).await;
                        }
                    };
                })
            }),
        );

        self.run_schedule(name.to_owned(), schedule).await
    }

    async fn run_schedule(&mut self, name: String, schedule: &str) {
        let func = self
            .job_registry
            .get_mut(&name)
            .expect("failed to get job")
            .clone();

        let job = Job::new_async(schedule, move |uuid, mut scheduler| {
            let name = name.clone();
            let func = func.clone();
            Box::pin(async move {
                debug!("Running job `{}`", name);

                (*func)().await;
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
    async fn new(public: bool, name: &str, service: Arc<dyn JobService>) -> Self {
        let id = if public {
            service
                .start(name.to_owned())
                .await
                .expect("failed to create job")
        } else {
            ID::from("".to_owned())
        };
        Self { id, service }
    }

    fn is_private(&self) -> bool {
        self.id.is_empty()
    }

    pub async fn stdout_writeline(&self, stdout: String) {
        if self.is_private() {
            return;
        }

        let stdout = stdout + "\n";
        match self.service.update_stdout(&self.id, stdout).await {
            Ok(_) => (),
            Err(_) => {
                warn!("Failed to write stdout to job `{}`", self.id);
            }
        }
    }

    pub async fn stderr_writeline(&self, stderr: String) {
        if self.is_private() {
            return;
        }

        let stderr = stderr + "\n";
        match self.service.update_stderr(&self.id, stderr).await {
            Ok(_) => (),
            Err(_) => {
                warn!("Failed to write stderr to job `{}`", self.id);
            }
        }
    }

    async fn complete(&self, exit_code: i32) {
        if self.is_private() {
            return;
        }

        match self.service.complete(&self.id, exit_code).await {
            Ok(_) => (),
            Err(_) => {
                warn!("Failed to complete job `{}`", self.id);
            }
        }
    }
}
