pub mod chat;
pub mod code;
pub mod completion;
pub mod event;
pub mod health;
pub mod model;

pub const DEFAULT_TEMPERATURE: f32 = 0.1;

pub fn default_seed() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64
}
