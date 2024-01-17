use std::error::Error;

use vergen::EmitBuilder;

fn main() -> Result<(), Box<dyn Error>> {
    if cfg!(feature = "rocm") {
        let rocm_root = "/opt/rocm";
        println!("cargo:rustc-link-search=native={}/lib", rocm_root);
    }
    // touch
    EmitBuilder::builder()
        .all_build()
        .all_git()
        .git_describe(true, true, Some("v*"))
        .emit()?;
    Ok(())
}
