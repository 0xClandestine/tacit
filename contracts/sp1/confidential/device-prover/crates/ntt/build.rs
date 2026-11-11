// Compiles the Metal host on macOS. Elsewhere the crate builds without it and reports the GPU path unavailable.
fn main() {
    println!("cargo:rerun-if-changed=src/metal_host.m");
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
        cc::Build::new().file("src/metal_host.m").flag("-fobjc-arc").compile("tacit_ntt_host");
        println!("cargo:rustc-link-lib=framework=Metal");
        println!("cargo:rustc-link-lib=framework=Foundation");
    }
}
