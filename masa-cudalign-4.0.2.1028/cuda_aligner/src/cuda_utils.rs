use cuda_runtime_sys as cuda;
use std::ptr;
use crate::CudaAlignerError;

pub unsafe fn cuda_malloc<T>(size: usize) -> Result<*mut T, CudaAlignerError> {
    let mut ptr = ptr::null_mut();
    match cuda::cudaMalloc(&mut ptr as *mut *mut _ as *mut *mut c_void, size) {
        cuda::cudaError_cudaSuccess => Ok(ptr as *mut T),
        err => Err(CudaAlignerError::MemoryError(
            format!("CUDA malloc failed with error: {}", err)
        ))
    }
}

pub unsafe fn cuda_memcpy<T>(
    dst: *mut T,
    src: *const T,
    count: usize,
    kind: cuda::cudaMemcpyKind,
) -> Result<(), CudaAlignerError> {
    match cuda::cudaMemcpy(
        dst as *mut c_void,
        src as *const c_void,
        count,
        kind,
    ) {
        cuda::cudaError_cudaSuccess => Ok(()),
        err => Err(CudaAlignerError::MemoryError(
            format!("CUDA memcpy failed with error: {}", err)
        ))
    }
}

pub unsafe fn cuda_memset<T>(
    devptr: *mut T,
    value: i32,
    count: usize,
) -> Result<(), CudaAlignerError> {
    match cuda::cudaMemset(devptr as *mut c_void, value, count) {
        cuda::cudaError_cudaSuccess => Ok(()),
        err => Err(CudaAlignerError::MemoryError(
            format!("CUDA memset failed with error: {}", err)
        ))
    }
}

pub unsafe fn cuda_free<T>(ptr: *mut T) -> Result<(), CudaAlignerError> {
    match cuda::cudaFree(ptr as *mut c_void) {
        cuda::cudaError_cudaSuccess => Ok(()),
        err => Err(CudaAlignerError::MemoryError(
            format!("CUDA free failed with error: {}", err)
        ))
    }
}