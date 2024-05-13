pub mod code;
pub mod doc;
pub mod event;
pub mod server_setting;

pub mod chat {
    use serde::{Deserialize, Serialize};
    use utoipa::ToSchema;

    #[derive(Serialize, Deserialize, ToSchema, Clone, Debug)]
    pub struct Message {
        pub role: String,
        pub content: String,
    }
}
