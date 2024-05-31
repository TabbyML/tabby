mod model;

mod chat;
pub use chat::create as create_chat;

mod completion;
pub use completion::create as create_completion;

mod embedding;
pub use embedding::create as create_embedding;
