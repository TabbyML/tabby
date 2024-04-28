mod controller;
mod db;
mod scheduler;

use std::sync::Arc;

use crate::schema::{
    auth::AuthenticationService, job::JobService, repository::RepositoryService,
    worker::WorkerService,
};

pub async fn run_cron(
    auth: Arc<dyn AuthenticationService>,
    job: Arc<dyn JobService>,
    worker: Arc<dyn WorkerService>,
    repository: Arc<dyn RepositoryService>,
    local_port: u16,
) {
    let mut controller = controller::JobController::new(job).await;
    db::register_jobs(
        &mut controller,
        auth,
        repository.github(),
        repository.gitlab(),
    )
    .await;

    scheduler::register(&mut controller, worker, local_port).await;

    controller.run().await
}
