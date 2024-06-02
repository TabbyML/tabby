use crate::layers::extensions::Data;
use crate::request::Request;
use crate::response::IntoResponse;
use futures::future::Map;
use futures::FutureExt;
use std::fmt;
use std::future::Future;
use std::marker::PhantomData;
use std::task::{Context, Poll};
use tower::Service;

/// A helper method to build functions
pub fn service_fn<T, K>(f: T) -> ServiceFn<T, K> {
    ServiceFn { f, k: PhantomData }
}

/// An executable service implemented by a closure.
///
/// See [`service_fn`] for more details.
#[derive(Copy, Clone)]
pub struct ServiceFn<T, K> {
    f: T,
    k: PhantomData<K>,
}

impl<T, K> fmt::Debug for ServiceFn<T, K> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ServiceFn")
            .field("f", &format_args!("{}", std::any::type_name::<T>()))
            .finish()
    }
}

/// The Future returned from [`ServiceFn`] service.
pub type FnFuture<F, O, R, E> = Map<F, fn(O) -> std::result::Result<R, E>>;

/// Allows getting some type from the [Request] data
pub trait FromData: Sized + Clone + Send + Sync + 'static {
    /// Gets the value
    fn get(data: &crate::extensions::Extensions) -> Self {
        data.get::<Self>().unwrap().clone()
    }
}

impl<T: Clone + Send + Sync + 'static> FromData for Data<T> {
    fn get(ctx: &crate::extensions::Extensions) -> Self {
        Data::new(ctx.get::<T>().unwrap().clone())
    }
}

macro_rules! impl_service_fn {
    ($($K:ident),+) => {
        #[allow(unused_parens)]
        impl<T, F, Req, E, R, $($K),+> Service<Request<Req>> for ServiceFn<T, ($($K),+)>
        where
            T: FnMut(Req, $($K),+) -> F,
            F: Future,
            F::Output: IntoResponse<Result = std::result::Result<R, E>>,
            $($K: FromData),+,
        {
            type Response = R;
            type Error = E;
            type Future = FnFuture<F, F::Output, R, E>;

            fn poll_ready(&mut self, _: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
                Poll::Ready(Ok(()))
            }

            fn call(&mut self, task: Request<Req>) -> Self::Future {
                let fut = (self.f)(task.req, $($K::get(&task.data)),+);

                fut.map(F::Output::into_response)
            }
        }
    };
}

impl<T, F, Req, E, R> Service<Request<Req>> for ServiceFn<T, ()>
where
    T: FnMut(Req) -> F,
    F: Future,
    F::Output: IntoResponse<Result = std::result::Result<R, E>>,
{
    type Response = R;
    type Error = E;
    type Future = FnFuture<F, F::Output, R, E>;

    fn poll_ready(&mut self, _: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        Poll::Ready(Ok(()))
    }

    fn call(&mut self, task: Request<Req>) -> Self::Future {
        let fut = (self.f)(task.req);

        fut.map(F::Output::into_response)
    }
}

impl_service_fn!(A);
impl_service_fn!(A1, A2);
impl_service_fn!(A1, A2, A3);
impl_service_fn!(A1, A2, A3, A4);
impl_service_fn!(A1, A2, A3, A4, A5);
impl_service_fn!(A1, A2, A3, A4, A5, A6);
impl_service_fn!(A1, A2, A3, A4, A5, A6, A7);
impl_service_fn!(A1, A2, A3, A4, A5, A6, A7, A8);
impl_service_fn!(A1, A2, A3, A4, A5, A6, A7, A8, A9);
impl_service_fn!(A1, A2, A3, A4, A5, A6, A7, A8, A9, A10);
impl_service_fn!(A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11);
impl_service_fn!(A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12);
impl_service_fn!(A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13);
impl_service_fn!(A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14);
impl_service_fn!(A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15);
impl_service_fn!(A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16);