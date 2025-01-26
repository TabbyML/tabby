use std::error::Error;

use vergen::EmitBuilder;

fn main() -> Result<(), Box<dyn Error>> {
    // touch
    EmitBuilder::builder()
        .all_build()
        .all_git()
        // TODO(kweizh): we encounter a issue with match_pattern in vergen on Windows
        // will add the match_pattern back when the issue is resolved
        // ref: https://github.com/rustyhorde/vergen/issues/402
        .git_describe(false, true, None)
        .emit()?;
    Ok(())
}
