use std::error::Error;

use vergen::EmitBuilder;

fn main() -> Result<(), Box<dyn Error>> {
    // touch
    EmitBuilder::builder().all_build().all_git().emit()?;
    Ok(())
}
