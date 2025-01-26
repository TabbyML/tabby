use std::error::Error;

use vergen_gitcl::{BuildBuilder, Emitter, GitclBuilder};

fn main() -> Result<(), Box<dyn Error>> {
    Emitter::default()
        .add_instructions(&BuildBuilder::all_build()?)?
        .add_instructions(
            &GitclBuilder::default()
                .all()
                .describe(false, true, Some("v*"))
                .build()?,
        )?
        .emit()?;
    Ok(())
}
