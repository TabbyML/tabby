use cmake::Config;

fn main() {
    println!("cargo:rerun-if-changed=cc/*.cc");

    Config::new(".")
        .define("LLAMA_METAL", "ON")
        .build();
}
