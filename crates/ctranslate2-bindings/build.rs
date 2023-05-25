use cmake::Config;

fn main() {
	let dst = Config::new("CTranslate2")
        // Default flags.
        .define("CMAKE_BUILD_TYPE", "Release")
        .define("BUILD_CLI", "OFF")
        .define("CMAKE_INSTALL_RPATH_USE_LINK_PATH", "ON")

        // FIXME(meng): support linux build.
        // OSX flags.
        .define("CMAKE_OSX_ARCHITECTURES", "arm64")
        .define("WITH_ACCELERATE", "ON")
        .define("WITH_MKL", "OFF")
        .define("OPENMP_RUNTIME", "NONE")
        .define("WITH_RUY", "ON")
        .build();

    println!("cargo:rustc-link-search=native={}", dst.join("lib").display());
    println!("cargo:rustc-link-lib=ctranslate2");

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
