use cuda_runtime_sys as cuda;

#[derive(Debug, Clone)]
pub struct OptimizationConfig {
    pub use_shared_memory: bool,
    pub use_texture_memory: bool,
    pub use_pinned_memory: bool,
    pub use_stream_processing: bool,
    pub block_pruning_threshold: f32,
}

impl Default for OptimizationConfig {
    fn default() -> Self {
        Self {
            use_shared_memory: true,
            use_texture_memory: true,
            use_pinned_memory: true,
            use_stream_processing: true,
            block_pruning_threshold: 0.1,
        }
    }
}

pub struct StreamProcessor {
    streams: Vec<cuda::cudaStream_t>,
    current_stream: usize,
}

impl StreamProcessor {
    pub fn new(num_streams: usize) -> Result<Self, CudaAlignerError> {
        let mut streams = Vec::with_capacity(num_streams);
        
        unsafe {
            for _ in 0..num_streams {
                let mut stream: cuda::cudaStream_t = std::ptr::null_mut();
                match cuda::cudaStreamCreate(&mut stream) {
                    cuda::cudaError_cudaSuccess => streams.push(stream),
                    err => return Err(CudaAlignerError::StreamError(
                        format!("Failed to create CUDA stream: {}", err)
                    )),
                }
            }
        }

        Ok(Self {
            streams,
            current_stream: 0,
        })
    }

    pub fn next_stream(&mut self) -> cuda::cudaStream_t {
        let stream = self.streams[self.current_stream];
        self.current_stream = (self.current_stream + 1) % self.streams.len();
        stream
    }
}

impl Drop for StreamProcessor {
    fn drop(&mut self) {
        unsafe {
            for stream in self.streams.drain(..) {
                cuda::cudaStreamDestroy(stream);
            }
        }
    }
} 