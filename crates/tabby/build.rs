use std::env;
use std::error::Error;

use vergen::EmitBuilder;

fn main() -> Result<(), Box<dyn Error>> {
    if cfg!(feature = "oneapi") {
        let mkl_root = env::var("MKLROOT")
            .expect("MKLROOT needs to be defined to compile for oneAPI (use setvars.sh to set)");
        let compiler_root = env::var("CMPLR_ROOT")
            .expect("CMPLR_ROOT needs to be defined to compile for oneAPI (use setvars.sh to set)");
        println!("cargo:rustc-link-arg=-fiopenmp");
        println!("cargo:rustc-link-arg=-fopenmp-targets=spir64_gen");
        println!("cargo:rustc-link-arg=-fsycl");
        println!("cargo:rustc-link-arg=-Wl,--no-as-needed");
        println!("cargo:rustc-link-search=native={}/lib", compiler_root);
        println!("cargo:rustc-link-search=native={}/lib", mkl_root);
        println!("cargo:rustc-link-lib=svml");
        println!("cargo:rustc-link-lib=mkl_sycl_blas");
        println!("cargo:rustc-link-lib=mkl_sycl_lapack");
        println!("cargo:rustc-link-lib=mkl_sycl_dft");
        println!("cargo:rustc-link-lib=mkl_sycl_sparse");
        println!("cargo:rustc-link-lib=mkl_sycl_vm");
        println!("cargo:rustc-link-lib=mkl_sycl_rng");
        println!("cargo:rustc-link-lib=mkl_sycl_stats");
        println!("cargo:rustc-link-lib=mkl_sycl_data_fitting");
        println!("cargo:rustc-link-lib=mkl_intel_ilp64");
        println!("cargo:rustc-link-lib=mkl_intel_thread");
        println!("cargo:rustc-link-lib=mkl_core");
        println!("cargo:rustc-link-lib=iomp5");
        println!("cargo:rustc-link-lib=sycl");
        println!("cargo:rustc-link-lib=pthread");
        println!("cargo:rustc-link-lib=m");
        println!("cargo:rustc-link-lib=dl");
    }
    // touch
    EmitBuilder::builder()
        .all_build()
        .all_git()
        .git_describe(true, true, Some("v*"))
        .emit()?;
    Ok(())
}
