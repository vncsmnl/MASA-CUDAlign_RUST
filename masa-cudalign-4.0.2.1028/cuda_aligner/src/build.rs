use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=src/kernels.cu");
    
    let cuda_path = std::env::var("CUDA_PATH")
        .expect("CUDA_PATH environment variable not set");
        
    cc::Build::new()
        .cuda(true)
        .flag("-cudart=shared")
        .flag(&format!("-I{}/include", cuda_path))
        .file("src/kernels.cu")
        .compile("cuda_kernels");
        
    println!("cargo:rustc-link-search=native={}/lib64", cuda_path);
    println!("cargo:rustc-link-lib=cudart");
}
