use cmake::Config;
use rust_cxx_cmake_bridge::read_cmake_generated;

fn main() {
    let mut config = Config::new(".");
    config
        .define("CMAKE_BUILD_TYPE", "Release")
        .define("BUILD_CLI", "OFF")
        .define("BUILD_SHARED_LIBS", "OFF")
        .define("CMAKE_INSTALL_RPATH_USE_LINK_PATH", "ON");

    if cfg!(target_os = "macos") {
        config
            .define("CMAKE_OSX_ARCHITECTURES", "arm64")
            .define("WITH_ACCELERATE", "ON")
            .define("WITH_MKL", "OFF")
            .define("OPENMP_RUNTIME", "NONE")
            .define("WITH_RUY", "ON");

        println!("cargo:rustc-link-lib=framework=Accelerate")
    } else if cfg!(target_os = "linux") {
        config
            .define("WITH_CUDA", "ON")
            .define("WITH_CUDNN", "ON")
            .define("WITH_MKL", "ON")
            .define("WITH_DNNL", "ON")
            .define("OPENMP_RUNTIME", "COMP")
            .cxxflag("-msse4.1")
            .define("CUDA_NVCC_FLAGS", "-Xfatbin=-compress-all")
            .define("CUDA_ARCH_LIST", "Common");
    }

    let dst = config.build();

    let cmake_generated_libs_str = std::fs::read_to_string(&format!("/{}/build/cmake_generated_libs", dst.display()).to_string()).unwrap();
    read_cmake_generated(&cmake_generated_libs_str);

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
