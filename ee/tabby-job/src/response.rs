use std::any::Any;

use crate::Error;

/// Helper for Job Responses
pub trait IntoResponse {
    /// The final result of the job
    type Result;
    /// converts self into a Result
    fn into_response(self) -> Self::Result;
}

impl IntoResponse for bool {
    type Result = std::result::Result<Self, Error>;
    fn into_response(self) -> std::result::Result<Self, Error> {
        match self {
            true => Ok(true),
            false => Err(Error::Failed(Box::new(std::io::Error::new(
                std::io::ErrorKind::Other,
                "Job returned false",
            )))),
        }
    }
}

impl<T: Any, E: std::error::Error + Sync + Send + 'static> IntoResponse
    for std::result::Result<T, E>
{
    type Result = Result<T, Error>;
    fn into_response(self) -> Result<T, Error> {
        match self {
            Ok(value) => Ok(value),
            Err(e) => Err(Error::Failed(Box::new(e))),
        }
    }
}

macro_rules! SIMPLE_JOB_RESULT {
    ($type:ty) => {
        impl IntoResponse for $type {
            type Result = std::result::Result<$type, Error>;
            fn into_response(self) -> std::result::Result<$type, Error> {
                Ok(self)
            }
        }
    };
}

SIMPLE_JOB_RESULT!(());
SIMPLE_JOB_RESULT!(u8);
SIMPLE_JOB_RESULT!(u16);
SIMPLE_JOB_RESULT!(u32);
SIMPLE_JOB_RESULT!(u64);
SIMPLE_JOB_RESULT!(usize);
SIMPLE_JOB_RESULT!(i8);
SIMPLE_JOB_RESULT!(i16);
SIMPLE_JOB_RESULT!(i32);
SIMPLE_JOB_RESULT!(i64);
SIMPLE_JOB_RESULT!(isize);
SIMPLE_JOB_RESULT!(f32);
SIMPLE_JOB_RESULT!(f64);
SIMPLE_JOB_RESULT!(String);
SIMPLE_JOB_RESULT!(&'static str);