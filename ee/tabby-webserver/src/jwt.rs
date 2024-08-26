use jsonwebtoken as jwt;
use juniper::ID;
use lazy_static::lazy_static;
use tabby_common::terminal::{HeaderFormat, InfoMessage};
use tabby_schema::auth::JWTPayload;
use tracing::warn;
use uuid::Uuid;

lazy_static! {
    static ref JWT_TOKEN_SECRET: String  = jwt_token_secret();

    static ref JWT_ENCODING_KEY: jwt::EncodingKey = jwt::EncodingKey::from_secret(
        JWT_TOKEN_SECRET.as_bytes()
    );
    static ref JWT_DECODING_KEY: jwt::DecodingKey = jwt::DecodingKey::from_secret(
        JWT_TOKEN_SECRET.as_bytes()
    );
    static ref JWT_DEFAULT_EXP: i64 = 30 * 60; // 30 minutes
}

pub fn generate_jwt(id: ID) -> jwt::errors::Result<String> {
    let claims = generate_jwt_payload(id, false);

    let header = jwt::Header::default();
    let token = jwt::encode(&header, &claims, &JWT_ENCODING_KEY)?;
    Ok(token)
}

pub fn generate_jwt_payload(id: ID, is_generated_from_auth_token: bool) -> JWTPayload {
    let iat = jwt::get_current_timestamp() as i64;
    let exp = iat + *JWT_DEFAULT_EXP;
    JWTPayload::new(id, iat, exp, is_generated_from_auth_token)
}

pub fn validate_jwt(token: &str) -> jwt::errors::Result<JWTPayload> {
    let validation = jwt::Validation::default();
    let data = jwt::decode::<JWTPayload>(token, &JWT_DECODING_KEY, &validation)?;
    Ok(data.claims)
}

fn jwt_token_secret() -> String {
    let jwt_secret = std::env::var("TABBY_WEBSERVER_JWT_TOKEN_SECRET").unwrap_or_else(|_| {
        InfoMessage::new("JWT secret is not set", HeaderFormat::BoldYellow, &[
            "Tabby server will generate a one-time (non-persisted) JWT secret for the current process.",
            &format!("Please set the {} environment variable for production usage.", HeaderFormat::Blue.format("TABBY_WEBSERVER_JWT_TOKEN_SECRET")),
        ]).print();
        Uuid::new_v4().to_string()
    });

    if Uuid::parse_str(&jwt_secret).is_err() {
        warn!("JWT token secret needs to be in standard uuid format to ensure its security, you might generate one at https://www.uuidgenerator.net");
        std::process::exit(1)
    }

    jwt_secret
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_generate_jwt() {
        let token = generate_jwt(ID::from("test".to_owned())).unwrap();

        assert!(!token.is_empty())
    }

    #[test]
    fn test_validate_jwt() {
        let token = generate_jwt(ID::from("test".to_owned())).unwrap();
        let claims = validate_jwt(&token).unwrap();
        assert_eq!(claims.sub.to_string(), "test");
    }
}
