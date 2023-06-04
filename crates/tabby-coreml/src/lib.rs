use objc::runtime::Object;
use objc::{class, msg_send, sel, sel_impl};

struct LanguageModel {}

impl LanguageModel {
    fn new() {
        unsafe {
            let class_nsstring = class!(NSString);
            let model_path: *mut Object = msg_send![class_nsstring, stringWithUTF8String: "file:///Users/meng/Projects/tabby/crates/tabby-coreml/cc/tiny-gptj"];

        }

        // let class_nsurl = class!(NSURL);
        // msg![class_nsurl, fileURLWithPath];
    }
}
