use std::{env, path::Path};

use cmake::Config;

fn main() {
    const LLAMA_CMAKE_PATH: &str = "llama.cpp/CMakeLists.txt";

    assert!(
        Path::new(LLAMA_CMAKE_PATH).exists(),
        "Please init submodules with `git submodule update --init --recursive` and try again"
    );

    println!("cargo:rerun-if-changed=include/engine.h");
    println!("cargo:rerun-if-changed=src/engine.cc");
    println!("cargo:rustc-link-lib=llama");
    println!("cargo:rustc-link-lib=ggml_static");

    build_llama_cpp();
    build_cxx_binding();
}

fn build_llama_cpp() {
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
        config.define("CMAKE_POSITION_INDEPENDENT_CODE", "ON");
        if cfg!(target_os = "windows") {
            let Ok(cuda_path) = env::var("CUDA_PATH") else {
                panic!("CUDA_PATH is not set");
            };
            println!(r"cargo:rustc-link-search=native={}\lib\x64", cuda_path);
        } else {
            println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
            println!("cargo:rustc-link-lib=culibos");
        }
        println!("cargo:rustc-link-lib=cudart");
        println!("cargo:rustc-link-lib=cublas");
        println!("cargo:rustc-link-lib=cublasLt");
    }
    if cfg!(feature = "rocm") {
        let amd_gpu_targets: Vec<&str> = vec![
            "gfx803",
            "gfx900",
            "gfx906:xnack-",
            "gfx908:xnack-",
            "gfx90a:xnack+",
            "gfx90a:xnack-",
            "gfx940",
            "gfx941",
            "gfx942",
            "gfx1010",
            "gfx1012",
            "gfx1030",
            "gfx1100",
            "gfx1101",
            "gfx1102",
            "gfx1103",
        ];

        let rocm_root = env::var("ROCM_ROOT").unwrap_or("/opt/rocm".to_string());
        config.define("LLAMA_HIPBLAS", "ON");
        config.define("CMAKE_C_COMPILER", format!("{}/llvm/bin/clang", rocm_root));
        config.define(
            "CMAKE_CXX_COMPILER",
            format!("{}/llvm/bin/clang++", rocm_root),
        );
        config.define("AMDGPU_TARGETS", amd_gpu_targets.join(";"));
        println!("cargo:rustc-link-arg=-Wl,--copy-dt-needed-entries");
        println!("cargo:rustc-link-search=native={}/hip/lib", rocm_root);
        println!("cargo:rustc-link-search=native={}/rocblas/lib", rocm_root);
        println!("cargo:rustc-link-search=native={}/hipblas/lib", rocm_root);
        println!("cargo:rustc-link-lib=amdhip64");
        println!("cargo:rustc-link-lib=rocblas");
        println!("cargo:rustc-link-lib=hipblas");
    }

    // By default, this value is automatically inferred from Rust’s compilation profile.
    // For Windows platform, we always build llama.cpp in release mode.
    // See https://github.com/TabbyML/tabby/pull/948 for more details.
    if cfg!(target_os = "windows") {
        config.profile("Release");
    }

    let dst = config.build();
    if cfg!(target_os = "windows") {
        println!(
            r"cargo:rustc-link-search=native={}\build\{}",
            dst.display(),
            config.get_profile()
        );
    } else {
        println!("cargo:rustc-link-search=native={}/build", dst.display());
    }
}

fn build_cxx_binding() {
    cxx_build::bridge("src/lib.rs")
        .file("src/engine.cc")
        .flag_if_supported("-Iinclude")
        .flag_if_supported("-Illama.cpp")
        .flag_if_supported("-std=c++14")
        .compile("cxxbridge");
}
