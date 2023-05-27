use std::path::PathBuf;
use cmake::Config;
use rust_cxx_cmake_bridge::read_cmake_generated;

fn main() {
    let dst = if cfg!(target_os = "macos") {
        ctranslate2_build_macos_static()
    } else if cfg!(target_os = "linux") {
        ctranslate2_build_linux_static()
    } else {
        panic!("Invalid target")
    };

    // Tell cargo to invalidate the built crate whenever the wrapper changes
    println!("cargo:rerun-if-changed=include/ctranslate2.h");
    println!("cargo:rerun-if-changed=src/ctranslate2.cc");
    println!("cargo:rerun-if-changed=src/lib.rs");

    cxx_build::bridge("src/lib.rs")
        .file("src/ctranslate2.cc")
        .flag_if_supported("-std=c++17")
        .flag_if_supported(&format!("-I{}", dst.join("include").display()))
        .compile("cxxbridge");
}

fn ctranslate2_build_linux_static() -> PathBuf {
    Config::new(".")
        .define("BUILD_CLI", "OFF")
        .define("CMAKE_INSTALL_RPATH_USE_LINK_PATH", "ON")
        .define("BUILD_SHARED_LIBS", "OFF")

        .define("BUILD_CLI", "OFF")
        .define("CMAKE_INSTALL_RPATH_USE_LINK_PATH", "ON")
        .define("WITH_CUDA", "ON")
        .define("WITH_CUDNN", "ON")
        .define("WITH_MKL", "ON")
        .define("WITH_DNNL", "ON")
        .define("OPENMP_RUNTIME", "COMP")
        .cxxflag("-msse4.1")
        .define("CUDA_NVCC_FLAGS", "-Xfatbin=-compress-all")
        .define("CUDA_ARCH_LIST", "Common")
        .build()
}

fn ctranslate2_build_macos_static() -> PathBuf {
    let dst = Config::new(".")
        .define("BUILD_CLI", "OFF")
        .define("CMAKE_INSTALL_RPATH_USE_LINK_PATH", "ON")
        .define("BUILD_SHARED_LIBS", "OFF")

        .define("CMAKE_OSX_ARCHITECTURES", "arm64")
        .define("WITH_ACCELERATE", "ON")
        .define("WITH_MKL", "OFF")
        .define("OPENMP_RUNTIME", "NONE")
        .define("WITH_RUY", "ON")
        .build();

    let cmake_generated_libs_str = std::fs::read_to_string(&format!("/{}/build/cmake_generated_libs", dst.display()).to_string()).unwrap();
    read_cmake_generated(&cmake_generated_libs_str);
    println!("cargo:rustc-link-lib=framework=Accelerate");

    return dst;
}
