use std::path::Path;

use cmake::Config;

fn main() {
    const LLAMA_CMAKE_PATH: &str = "llama.cpp/CMakeLists.txt";

    assert!(
        Path::new(LLAMA_CMAKE_PATH).exists(),
        "Please init submodules with `git submodule update --init --recursive` and try again"
    );

    println!("cargo:rerun-if-changed=cc/*.h");
    println!("cargo:rerun-if-changed=cc/*.cc");
    println!("cargo:rustc-link-lib=llama");
    println!("cargo:rustc-link-lib=ggml_static");

    let mut config = Config::new("llama.cpp");
    if cfg!(target_os = "macos") {
        config.define("LLAMA_METAL", "ON");
        println!("cargo:rustc-link-lib=framework=Foundation");
        println!("cargo:rustc-link-lib=framework=Accelerate");
        println!("cargo:rustc-link-lib=framework=Metal");
        println!("cargo:rustc-link-lib=framework=MetalKit");
    }
    if cfg!(feature = "cuda") {
        config.define("LLAMA_CUBLAS", "ON");
        println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
        println!("cargo:rustc-link-lib=cudart");
        println!("cargo:rustc-link-lib=culibos");
        println!("cargo:rustc-link-lib=cublas");
        println!("cargo:rustc-link-lib=cublasLt");
    }
    if cfg!(feature = "rocm") {
        config.define("LLAMA_HIPBLAS", "ON");
        config.define("CMAKE_C_COMPILER", "/opt/rocm/llvm/bin/clang");
        config.define("CMAKE_CXX_COMPILER", "/opt/rocm/llvm/bin/clang++");
        println!("cargo:rustc-link-search=native=/opt/rocm/hipblas/lib");
        println!("cargo:rustc-link-search=native=/opt/rocm/hipblaslt/lib");
        println!("cargo:rustc-link-lib=hipblas");
        println!("cargo:rustc-link-lib=hipblaslt");
    }
    if cfg!(feature = "oneapi") {
        config.define("LLAMA_BLAS", "ON");
        config.define("LLAMA_BLAS_VENDOR", "Intel10_64lp");
        println!("cargo:rustc-link-search=native=/opt/intel/oneapi/mkl/latest/lib");
        println!("cargo:rustc-link-lib=mkl_rt");
        println!("cargo:rustc-link-lib=pthread");
        println!("cargo:rustc-link-lib=m");
        println!("cargo:rustc-link-lib=dl");
    }

    let dst = config.build();
    println!("cargo:rustc-link-search=native={}/build", dst.display());

    cxx_build::bridge("src/lib.rs")
        .file("src/engine.cc")
        .flag_if_supported("-Iinclude")
        .flag_if_supported("-Illama.cpp")
        .flag_if_supported("-std=c++14")
        .compile("cxxbridge");
}
