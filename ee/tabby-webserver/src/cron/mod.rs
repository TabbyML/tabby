mod controller;
mod db;
mod scheduler;

use std::sync::Arc;

use rand::Rng;
use tabby_db::DbConn;

use crate::schema::{
    auth::AuthenticationService, repository::RepositoryService, worker::WorkerService,
};

#[macro_export]
macro_rules! warn_stderr {
    ($ctx:expr, $($params:tt)+) => {
        tracing::warn!($($params)+);
        $ctx.stderr_writeline(format!($($params)+)).await;
    }
}

pub async fn run_cron(
    schedule_event_sender: tokio::sync::mpsc::UnboundedSender<String>,
    schedule_event_receiver: tokio::sync::mpsc::UnboundedReceiver<String>,
    db: DbConn,
    auth: Arc<dyn AuthenticationService>,
    worker: Arc<dyn WorkerService>,
    repository: Arc<dyn RepositoryService>,
    local_port: u16,
) {
    let mut controller = controller::JobController::new(db, schedule_event_sender).await;
    db::register_jobs(
        &mut controller,
        auth,
        repository.github(),
        repository.gitlab(),
    )
    .await;

    scheduler::register(&mut controller, worker, local_port).await;

    let controller = Arc::new(controller);
    controller.start_worker(schedule_event_receiver);
    controller.start_cron().await
}

fn every_two_hours() -> String {
    let mut rng = rand::thread_rng();
    format!(
        "{} {} */2 * * *",
        rng.gen_range(0..59),
        rng.gen_range(0..59)
    )
}

fn every_ten_minutes() -> String {
    let mut rng = rand::thread_rng();
    format!("{} */10 * * * *", rng.gen_range(0..59))
}
