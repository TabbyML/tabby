use juniper::{GraphQLEnum, GraphQLObject};
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

#[derive(Serialize, Deserialize, GraphQLEnum, ToSchema, PartialEq, Clone, Debug)]
pub enum DeviceType {
    Cuda,
    Rocm,
}

#[derive(Serialize, Deserialize, GraphQLObject, ToSchema, Clone, Debug)]
pub struct Accelerator {
    /// Universally unique ID of the accelerator, if available
    pub uuid: Option<String>,
    /// Technical name of the underlying hardware chip, if available
    pub chip_name: Option<String>,
    /// User readable name for the accelerator
    pub display_name: String,
    /// Type of the accelerator device
    pub device_type: DeviceType,
}
