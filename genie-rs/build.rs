use std::env;
use std::path::PathBuf;
fn main() {
    let qnn_sdk_root = env::var("QAIRT").unwrap_or_else(|_| {
        let home = env::var("HOME").expect("HOME not set");
        format!("{}/qairt/2.47.0.260601", home)
    });
    let lib_dir = format!("{}/lib/aarch64-oe-linux-gcc11.2", qnn_sdk_root);
    let include_dir = format!("{}/include", qnn_sdk_root);
    println!("cargo:rustc-link-search=native={}", lib_dir);
    println!("cargo:rustc-link-lib=dylib=Genie");
    println!("cargo:rustc-link-lib=dylib=QnnHtp");
    println!("cargo:rustc-link-lib=dylib=QnnSystem");
    // Embed rpath: try test model dir first (has working libQnnHtp.so), then SDK
    let llama_model_lib = format!("{}/llama-v68-model", env::var("HOME").expect("HOME not set"));
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", llama_model_lib);
    let test_model_lib = format!("{}/Qwen2.5-0.5B-v68", env::var("HOME").expect("HOME not set"));
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", test_model_lib);
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib_dir);
    let bindings = bindgen::Builder::default()
        .header(format!("{}/Genie/GenieCommon.h", include_dir))
        .header(format!("{}/Genie/GenieDialog.h", include_dir))
        .header(format!("{}/Genie/GenieTokenizer.h", include_dir))
        .header(format!("{}/Genie/GenieSampler.h", include_dir))
        .header(format!("{}/Genie/GenieLog.h", include_dir))
        .header(format!("{}/Genie/GenieProfile.h", include_dir))
        .clang_arg(format!("-I{}", include_dir))
        .allowlist_function("Genie.*")
        .allowlist_type("Genie.*")
        .allowlist_var("GENIE.*")
        .generate()
        .expect("Unable to generate bindings");
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("genie_bindings.rs"))
        .expect("Couldn't write bindings!");
}
