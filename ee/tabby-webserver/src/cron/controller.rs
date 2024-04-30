use std::{collections::HashMap, pin::Pin, sync::Arc};

use chrono::Utc;
use futures::Future;
use tabby_db::DbConn;
use tokio::sync::mpsc::{UnboundedReceiver, UnboundedSender};
use tokio_cron_scheduler::{Job, JobScheduler};
use tracing::{debug, info, warn};

pub struct JobController {
    scheduler: JobScheduler,
    db: DbConn,
    job_registry: HashMap<
        &'static str,
        Arc<dyn Fn() -> Pin<Box<dyn Future<Output = ()> + Send>> + Send + Sync + 'static>,
    >,
    event_sender: UnboundedSender<String>,
}

impl JobController {
    pub async fn new(db: DbConn, event_sender: UnboundedSender<String>) -> Self {
        db.finalize_stale_job_runs()
            .await
            .expect("failed to cleanup stale jobs");
        let scheduler = JobScheduler::new()
            .await
            .expect("failed to create job scheduler");
        Self {
            scheduler,
            db,
            job_registry: HashMap::default(),
            event_sender,
        }
    }

    fn run_job(&self, name: &str) -> tokio::task::JoinHandle<()> {
        let func = self
            .job_registry
            .get(name)
            .expect("failed to get job")
            .clone();

        // Spawn a new thread for panic isolation
        tokio::task::spawn(async move {
            func().await;
        })
    }

    /// Start the worker that listens for job events and runs the jobs.
    ///
    /// 1. Only one instance of the job will be run at a time.
    /// 2. Jobs are deduplicated within a time window (120 seconds).
    pub fn start_worker(self: &Arc<Self>, mut event_receiver: UnboundedReceiver<String>) {
        const JOB_DEDUPE_WINDOW_SECS: i64 = 120;
        let controller = self.clone();
        tokio::spawn(async move {
            // Sleep for 5 seconds to allow the webserver to start.
            tokio::time::sleep(tokio::time::Duration::from_secs(5)).await;

            let mut last_timestamps = HashMap::new();
            loop {
                while let Some(name) = event_receiver.recv().await {
                    if let Some(last_timestamp) = last_timestamps.get(&name) {
                        if Utc::now()
                            .signed_duration_since(*last_timestamp)
                            .num_seconds()
                            < JOB_DEDUPE_WINDOW_SECS
                        {
                            info!("Job `{name}` last ran less than {JOB_DEDUPE_WINDOW_SECS} seconds ago (@{last_timestamp}), skipped");
                            continue;
                        }
                    }

                    last_timestamps.insert(name.clone(), Utc::now());
                    let _ = controller.run_job(&name).await;
                }
            }
        });
    }

    pub async fn start_cron(&self) {
        if std::env::var("TABBY_WEBSERVER_CONTROLLER_ONESHOT").is_ok() {
            warn!(
            "Running controller job as oneshot, this should only be used for debugging purpose..."
        );
            for name in self.job_registry.keys() {
                let _ = self.event_sender.send(name.to_string());
            }
        } else {
            self.scheduler
                .start()
                .await
                .expect("failed to start job scheduler")
        }
    }

    /// Register a new job with the scheduler, the job will be displayed in Jobs dashboard.
    pub async fn register_public<T>(&mut self, name: &'static str, schedule: &str, func: T)
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
    pub async fn register<T>(&mut self, name: &'static str, schedule: &str, func: T)
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

    async fn register_impl<F>(
        &mut self,
        is_public: bool,
        name: &'static str,
        schedule: &str,
        func: F,
    ) where
        F: FnMut(&JobContext) -> Pin<Box<dyn Future<Output = anyhow::Result<i32>> + Send>>
            + Send
            + Sync
            + Clone
            + 'static,
    {
        let db = self.db.clone();
        self.job_registry.insert(
            name,
            Arc::new(move || {
                let db = db.clone();
                let mut func = func.clone();

                Box::pin(async move {
                    debug!("Running job `{}`", name);
                    let context = JobContext::new(is_public, name, db.clone()).await;
                    match func(&context).await {
                        Ok(exit_code) => {
                            debug!("Job `{}` completed with exit code {}", name, exit_code);
                            context.complete(exit_code).await;
                        }
                        Err(e) => {
                            warn!("Job `{}` failed: {}", name, e);
                            context.complete(-1).await;
                        }
                    };
                })
            }),
        );

        self.add_to_schedule(name, schedule).await
    }

    async fn add_to_schedule(&mut self, name: &'static str, schedule: &str) {
        let event_sender = self.event_sender.clone();
        let job = Job::new_async(schedule, move |uuid, mut scheduler| {
            let event_sender = event_sender.clone();
            Box::pin(async move {
                if let Err(err) = event_sender.send(name.to_owned()) {
                    warn!("Failed to schedule job `{}`: {}", &name, err);
                } else {
                    debug!("Scheduling job `{}`", &name);
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
    id: i64,
    db: DbConn,
}

impl JobContext {
    async fn new(public: bool, name: &'static str, db: DbConn) -> Self {
        let id = if public {
            db.create_job_run(name.to_owned())
                .await
                .expect("failed to create job")
        } else {
            -1
        };
        Self { id, db }
    }

    fn is_private(&self) -> bool {
        self.id < 0
    }

    pub async fn stdout_writeline(&self, stdout: String) {
        if self.is_private() {
            return;
        }

        let stdout = stdout + "\n";
        match self.db.update_job_stdout(self.id, stdout).await {
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
        match self.db.update_job_stderr(self.id, stderr).await {
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

        match self.db.update_job_status(self.id, exit_code).await {
            Ok(_) => (),
            Err(_) => {
                warn!("Failed to complete job `{}`", self.id);
            }
        }
    }
}
