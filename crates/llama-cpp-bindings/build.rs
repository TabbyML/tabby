use cmake::Config;

fn main() {
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
    if cfg!(feature = "opencl") {
        config.define("LLAMA_CLBLAST", "ON");
        println!("cargo:rustc-link-search=native=/opt/intel/opencl/intel64/");        
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
