#[tarpc::service]
pub trait WebserverApi {
    async fn hello();
}
