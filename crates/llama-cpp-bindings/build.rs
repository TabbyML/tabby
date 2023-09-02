use cmake::Config;

fn main() {
    Config::new("llama.cpp")
        .define("LLAMA_METAL", "ON")
        .build();

    println!("cargo:rerun-if-changed=engine/*.h");
    println!("cargo:rerun-if-changed=engine/*.cc");
    cxx_build::bridge("bridge/lib.rs")
        .file("bridge/engine.cc")
        .flag_if_supported("-Illama.cpp")
        .flag_if_supported("-std=c++14")
        .compile("cxxbridge");
}
