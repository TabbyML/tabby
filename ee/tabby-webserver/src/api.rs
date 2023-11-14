use crate::schema::WorkerKind;

#[tarpc::service]
pub trait WebserverApi {
    async fn hello();
}