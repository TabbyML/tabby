use clap::{Args, Parser, Subcommand};

use crate::Device;

#[derive(Args)]
pub struct ModelArgs {
    /// Model id for `/completions` API endpoint.
    #[clap(long)]
    model: String,

    #[clap(long, default_value_t = 8080)]
    port: u16,

    /// Device to run model inference.
    #[clap(long, default_value_t=Device::Cpu)]
    device: Device,

    /// Parallelism for model serving - increasing this number will have a significant impact on the
    /// memory requirement e.g., GPU vRAM.
    #[clap(long, default_value_t = 1)]
    parallelism: u8,
}

type CompletionArgs = ModelArgs;
type ChatArgs = ModelArgs;

#[derive(Parser)]
pub struct WorkerArgs {
    #[structopt(subcommand)]
    pub worker_commands: WorkerCommands,
}

#[derive(Subcommand)]
pub enum WorkerCommands {
    /// Start completion worker.
    Completion(CompletionArgs),

    /// Start chat worker.
    Chat(ChatArgs),
}

pub async fn main(args: &WorkerArgs) {}