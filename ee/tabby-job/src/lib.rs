mod request;
mod response;

mod extensions;
mod layers;
mod service_fn;

use thiserror::Error;

#[derive(Error, Debug)]
pub enum Error {
    #[error("Task Failed: {0}")]
    Failed(#[source] BoxDynError),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

pub type BoxDynError = Box<dyn std::error::Error + 'static + Send + Sync>;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::request::Request;

    use assert_matches::assert_matches;
    use layers::extensions::Data;
    use tower::{Service, ServiceBuilder, ServiceExt};

    struct TestJob;
    async fn test_handler(_request: TestJob, data: Data<i32>) -> Result<i32, Error> {
        Ok(*data + 2)
    }

    #[tokio::test]
    async fn it_works() -> Result<(), Error> {
        let mut service = ServiceBuilder::new()
            .layer(Data::new(40))
            .concurrency_limit(1)
            .service(service_fn::service_fn(test_handler));
        let response = service.ready().await?.call(Request::new(TestJob)).await;
        assert_matches!(response, Ok(42));

        Ok(())
    }
}
