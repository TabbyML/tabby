use crate::schema::{Worker, WorkerKind};

#[tarpc::service]
pub trait WebserverApi {
    async fn register_worker_as(
        kind: WorkerKind,
        port: i32,
        name: String,
        device: String,
        arch: String,
        cpu_info: String,
        cpu_count: i32,
        cuda_devices: Vec<String>,
    ) -> Worker;
}
