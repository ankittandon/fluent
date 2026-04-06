fn main() {
    let mut build = cc::Build::new();
    build
        .cpp(true)
        .std("c++17")
        .file("cpp/speaker_features.cpp")
        .opt_level(3);

    // On macOS, use Accelerate framework for hardware-optimised FFT.
    if cfg!(target_os = "macos") {
        build.flag("-DACCELERATE_NEW_LAPACK");
        println!("cargo:rustc-link-lib=framework=Accelerate");
    }

    build.compile("screamer_speaker_features");

    println!("cargo:rerun-if-changed=cpp/speaker_features.cpp");
    println!("cargo:rerun-if-changed=cpp/speaker_features.h");
}
