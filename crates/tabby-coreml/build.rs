fn main() {
    cc::Build::new()
        .file("cc/lib.mm")
        .compile("tabby-coreml");
}
