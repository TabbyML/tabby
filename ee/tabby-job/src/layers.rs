pub mod extensions {
    use std::{
        ops::Deref,
        task::{Context, Poll},
    };
    use tower::Service;

    use crate::request::Request;

    /// Extension data for tasks.
    /// This is commonly used to share state across tasks. or across layers within the same tasks
    ///
    /// ```rust
    /// # use std::sync::Arc;
    /// # struct Email;
    /// # use apalis_core::layers::extensions::Data;
    /// # use apalis_core::service_fn::service_fn;
    /// # use crate::apalis_core::builder::WorkerFactory;
    /// # use apalis_core::builder::WorkerBuilder;
    /// # use apalis_core::memory::MemoryStorage;
    /// // Some shared state used throughout our application
    /// struct State {
    ///     // ...
    /// }
    ///
    /// async fn email_service(email: Email, state: Data<Arc<State>>) {
    ///     
    /// }
    ///
    /// let state = Arc::new(State { /* ... */ });
    ///
    /// let worker = WorkerBuilder::new("tasty-avocado")
    ///     .data(state)
    ///     .source(MemoryStorage::new())
    ///     .build(service_fn(email_service));
    /// ```

    #[derive(Debug, Clone, Copy)]
    pub struct Data<T>(T);
    impl<T> Data<T> {
        /// Build a new data entry
        pub fn new(inner: T) -> Data<T> {
            Data(inner)
        }
    }

    impl<T> Deref for Data<T> {
        type Target = T;
        fn deref(&self) -> &Self::Target {
            &self.0
        }
    }

    impl<S, T> tower::Layer<S> for Data<T>
    where
        T: Clone + Send + Sync + 'static,
    {
        type Service = AddExtension<S, T>;

        fn layer(&self, inner: S) -> Self::Service {
            AddExtension {
                inner,
                value: self.0.clone(),
            }
        }
    }

    /// Middleware for adding some shareable value to [request data].
    #[derive(Clone, Copy, Debug)]
    pub struct AddExtension<S, T> {
        inner: S,
        value: T,
    }

    impl<S, T, Req> Service<Request<Req>> for AddExtension<S, T>
    where
        S: Service<Request<Req>>,
        T: Clone + Send + Sync + 'static,
    {
        type Response = S::Response;
        type Error = S::Error;
        type Future = S::Future;

        #[inline]
        fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
            self.inner.poll_ready(cx)
        }

        fn call(&mut self, mut req: Request<Req>) -> Self::Future {
            req.data.insert(self.value.clone());
            self.inner.call(req)
        }
    }
}