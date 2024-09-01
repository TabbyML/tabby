use axum::http::HeaderName;
use axum_extra::headers::Header;

use crate::constants::USER_HEADER_FIELD_NAME;

#[derive(Debug)]
pub struct MaybeUser(pub Option<String>);

pub static USER_HEADER: HeaderName = HeaderName::from_static(USER_HEADER_FIELD_NAME);

impl Header for MaybeUser {
    fn name() -> &'static axum::http::HeaderName {
        &USER_HEADER
    }

    fn decode<'i, I>(values: &mut I) -> Result<Self, axum_extra::headers::Error>
    where
        Self: Sized,
        I: Iterator<Item = &'i axum::http::HeaderValue>,
    {
        let Some(value) = values.next() else {
            return Ok(MaybeUser(None));
        };
        let str = value.to_str().expect("User email is always a valid string");
        Ok(MaybeUser(Some(str.to_string())))
    }

    fn encode<E: Extend<axum::http::HeaderValue>>(&self, _values: &mut E) {
        todo!()
    }
}
