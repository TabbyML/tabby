use std::{env, path::Path};

use cmake::Config;
use omnicopy_to_output::copy_to_output;

fn main() {
    if !cfg!(feature = "binary") || env::var("CI_COVERAGE").is_ok() {
        return;
    }

    let mut config = Config::new("./llama.cpp");
    config.profile("Release");
    config.define("GGML_NATIVE", "OFF");
    config.define("GGML_NATIVE_DEFAULT", "OFF");
    config.define("BUILD_SHARED_LIBS", "OFF");

    if cfg!(target_os = "macos") {
        config.define("LLAMA_METAL", "ON");
        config.define("LLAMA_METAL_EMBED_LIBRARY", "ON");
        println!("cargo:rustc-link-lib=framework=Foundation");
        println!("cargo:rustc-link-lib=framework=Accelerate");
        println!("cargo:rustc-link-lib=framework=Metal");
        println!("cargo:rustc-link-lib=framework=MetalKit");
    }
    if cfg!(feature = "cuda") {
        config.define("GGML_CUDA", "ON");
        config.define("CMAKE_POSITION_INDEPENDENT_CODE", "ON");
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
            "gfx1031",
            "gfx1100",
            "gfx1101",
            "gfx1102",
            "gfx1103",
        ];

        let rocm_root = env::var("ROCM_ROOT").unwrap_or("/opt/rocm".to_string());
        config.define("GGML_HIPBLAS", "ON");
        config.define("CMAKE_C_COMPILER", format!("{}/llvm/bin/clang", rocm_root));
        config.define(
            "CMAKE_CXX_COMPILER",
            format!("{}/llvm/bin/clang++", rocm_root),
        );
        config.define("AMDGPU_TARGETS", amd_gpu_targets.join(";"));
    }
    if cfg!(feature = "vulkan") {
        config.define("LLAMA_VULKAN", "ON");
    }

    let out = config.build();
    let server_binary = make_output_binary(&out, "llama-server");

    copy_to_output(&server_binary).expect("Failed to copy server binary to output directory");
}

fn make_output_binary(out: &Path, name: &str) -> String {
    out.join("bin").join(name).display().to_string() + env::consts::EXE_SUFFIX
}
