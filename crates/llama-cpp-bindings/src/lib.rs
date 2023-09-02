#[cxx::bridge(namespace = "tabby")]
mod ffi {
    unsafe extern "C++" {
        include!("llama-cpp-bindings/cc/engine.h");

        type LlamaEngine;

        fn create_engine(model_path: &str) -> SharedPtr<LlamaEngine>;

        fn inference(
            &self,
            tokens: &str,
            max_decoding_length: usize,
            sampling_temperature: f32,
        ) -> Vec<u32>;
    }
}


unsafe impl Send for ffi::LlamaEngine {}
unsafe impl Sync for ffi::LlamaEngine {}
