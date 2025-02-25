use std::ffi::c_void;
use std::ptr;
use cuda_runtime_sys as cuda;
mod cuda_utils;
use cuda_utils::*;
use std::time::Instant;
use crate::scoring::{AlignmentScore, Position, AlignmentStats, BlockScores};
use crate::alignment_mode::{AlignmentMode, AlignmentParams};
use crate::traceback::{Traceback, TracebackMatrix};
use crate::optimizations::{OptimizationConfig, StreamProcessor};

// Constants from the original code
pub const THREADS_COUNT: i32 = 128;
pub const MAX_BLOCKS_COUNT: i32 = 512;
pub const ALPHA: i32 = 4;
pub const MAX_GRID_HEIGHT: i32 = THREADS_COUNT * MAX_BLOCKS_COUNT;
pub const MAX_SEQUENCE_SIZE: i32 = 134_150_000;

// DNA scoring parameters
pub const DNA_MATCH: i32 = 1;
pub const DNA_MISMATCH: i32 = -3;
pub const DNA_GAP_OPEN: i32 = 3;
pub const DNA_GAP_EXT: i32 = 2;
pub const DNA_GAP_FIRST: i32 = DNA_GAP_EXT + DNA_GAP_OPEN;

#[derive(Debug, Error)]
pub enum CudaAlignerError {
    #[error("CUDA initialization failed: {0}")]
    InitializationError(String),
    #[error("Sequence too large: {0}")]
    SequenceTooLarge(usize),
    #[error("Memory allocation failed: {0}")]
    MemoryError(String),
    #[error("Invalid GPU device: {0}")]
    InvalidDevice(i32),
    #[error("Kernel error: {0}")]
    KernelError(String),
    #[error("Texture error: {0}")]
    TextureError(String),
    #[error("No results found")]
    NoResults,
}

// Equivalent to the original score_params_t
#[derive(Debug, Clone, Copy)]
pub struct ScoreParams {
    pub match_score: i32,
    pub mismatch: i32,
    pub gap_open: i32,
    pub gap_ext: i32,
}

impl Default for ScoreParams {
    fn default() -> Self {
        Self {
            match_score: DNA_MATCH,
            mismatch: DNA_MISMATCH,
            gap_open: DNA_GAP_OPEN,
            gap_ext: DNA_GAP_EXT,
        }
    }
}

// Equivalent to the original CUDAlignerParameters
#[derive(Debug)]
pub struct CudaAlignerParameters {
    gpu: i32,
    blocks: i32,
}

impl Default for CudaAlignerParameters {
    fn default() -> Self {
        Self {
            gpu: -1, // DETECT_FASTEST_GPU
            blocks: 0,
        }
    }
}

impl CudaAlignerParameters {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_gpu(mut self, gpu: i32) -> Self {
        self.gpu = gpu;
        self
    }

    pub fn with_blocks(mut self, blocks: i32) -> Result<Self, CudaAlignerError> {
        if blocks > MAX_BLOCKS_COUNT {
            return Err(CudaAlignerError::InitializationError(
                format!("Blocks count cannot be greater than {}", MAX_BLOCKS_COUNT)
            ));
        }
        self.blocks = blocks;
        Ok(self)
    }

    pub fn gpu(&self) -> i32 {
        self.gpu
    }

    pub fn blocks(&self) -> i32 {
        self.blocks
    }
}

// Safe wrapper for CUDA memory structures
#[derive(Default)]
pub struct HostStructures {
    h_bus_h: Option<*mut cuda::int2>,
    h_extra_h: Option<*mut cuda::int2>,
    h_flush_column_h: Option<*mut cuda::int4>,
    h_flush_column_e: Option<*mut cuda::int4>,
    h_flush_column: Option<*mut Cell>,
    h_load_column_h: Option<*mut cuda::int4>,
    h_load_column_e: Option<*mut cuda::int4>,
    h_block_result: Option<*mut cuda::int4>,
    h_block_scores: Option<*mut Score>,
}

#[derive(Default)]
pub struct CudaStructures {
    d_seq0: Option<*mut u8>,
    d_seq1: Option<*mut u8>,
    d_bus_h: Option<*mut cuda::int2>,
    d_extra_h: Option<*mut cuda::int2>,
    d_flush_column_h: Option<*mut cuda::int4>,
    d_flush_column_e: Option<*mut cuda::int4>,
    d_load_column_h: Option<*mut cuda::int4>,
    d_load_column_e: Option<*mut cuda::int4>,
    d_block_result: Option<*mut cuda::int4>,
    d_bus_v_h: Option<*mut cuda::int4>,
    d_bus_v_e: Option<*mut cuda::int4>,
    d_bus_v_o: Option<*mut cuda::int3>,
    bus_h_size: i32,
    d_traceback_matrix: Option<*mut u8>,
}

// Main CudaAligner struct
pub struct CudaAligner {
    host: HostStructures,
    cuda: CudaStructures,
    multiprocessors: i32,
    params: CudaAlignerParameters,
    score_params: ScoreParams,
    alignment_params: AlignmentParams,
    
    // Statistics
    stat_total_mem: usize,
    stat_initial_used_memory: usize,
    stat_fixed_allocated_memory: usize,
    stat_variable_allocated_memory: usize,
    stat_deallocated_memory: usize,
    block_scores: BlockScores,
    stats_buffer: Option<*mut cuda::int4>,
    optimization_config: OptimizationConfig,
    stream_processor: Option<StreamProcessor>,
}

// Add new sequence-related structures
#[derive(Debug, Clone)]
pub struct Sequence {
    data: Vec<u8>,
    length: usize,
}

impl Sequence {
    pub fn new(data: &[u8]) -> Self {
        Self {
            data: data.to_vec(),
            length: data.len(),
        }
    }
}

impl CudaAligner {
    pub fn new(params: CudaAlignerParameters) -> Result<Self, CudaAlignerError> {
        Ok(Self {
            host: HostStructures::default(),
            cuda: CudaStructures::default(),
            multiprocessors: 0,
            params,
            score_params: ScoreParams::default(),
            alignment_params: AlignmentParams::default(),
            stat_total_mem: 0,
            stat_initial_used_memory: 0,
            stat_fixed_allocated_memory: 0,
            stat_variable_allocated_memory: 0,
            stat_deallocated_memory: 0,
            block_scores: BlockScores::new(params.blocks() as usize),
            stats_buffer: None,
            optimization_config: OptimizationConfig::default(),
            stream_processor: None,
        })
    }

    pub fn initialize(&mut self) -> Result<(), CudaAlignerError> {
        // Initialize CUDA device
        self.select_gpu()?;
        self.check_cuda_capability()?;
        self.allocate_global_structures()?;
        Ok(())
    }

    fn select_gpu(&self) -> Result<(), CudaAlignerError> {
        let gpu_id = if self.params.gpu() == -1 {
            // TODO: Implement fastest GPU detection
            0
        } else {
            self.params.gpu()
        };

        unsafe {
            // Set CUDA device
            match cuda::cudaSetDevice(gpu_id) {
                cuda::cudaError_cudaSuccess => Ok(()),
                err => Err(CudaAlignerError::InvalidDevice(err)),
            }
        }
    }

    fn check_cuda_capability(&self) -> Result<(), CudaAlignerError> {
        // TODO: Implement CUDA capability checking
        Ok(())
    }

    fn allocate_global_structures(&mut self) -> Result<(), CudaAlignerError> {
        unsafe {
            // Allocate host memory
            self.host.h_extra_h = Some(
                std::alloc::alloc(
                    std::alloc::Layout::array::<cuda::int2>(THREADS_COUNT as usize)?
                ) as *mut cuda::int2
            );
            
            self.host.h_flush_column_h = Some(
                std::alloc::alloc(
                    std::alloc::Layout::array::<cuda::int4>(THREADS_COUNT as usize)?
                ) as *mut cuda::int4
            );
            
            // ... (similar allocations for other host structures)

            // Allocate CUDA memory
            self.cuda.d_extra_h = Some(cuda_malloc(
                THREADS_COUNT as usize * std::mem::size_of::<cuda::int2>()
            )?);

            self.cuda.d_flush_column_h = Some(cuda_malloc(
                THREADS_COUNT as usize * std::mem::size_of::<cuda::int4>()
            )?);

            // ... (similar allocations for other CUDA structures)

            // Initialize statistics
            self.stat_fixed_allocated_memory = self.get_memory_usage()? - self.stat_initial_used_memory;
            self.stat_deallocated_memory = self.stat_fixed_allocated_memory;

            Ok(())
        }
    }

    pub fn set_sequences(
        &mut self,
        seq0: &Sequence,
        seq1: &Sequence,
    ) -> Result<(), CudaAlignerError> {
        if seq0.length == 0 || seq1.length == 0 {
            return Ok(());
        }

        if seq0.length > MAX_SEQUENCE_SIZE as usize || seq1.length > MAX_SEQUENCE_SIZE as usize {
            return Err(CudaAlignerError::SequenceTooLarge(
                std::cmp::max(seq0.length, seq1.length)
            ));
        }

        let seq0_padding = MAX_GRID_HEIGHT as usize;
        let seq1_padding = 0;
        let bus_size = (seq1.length + seq1_padding + 1) * std::mem::size_of::<cuda::int2>();

        unsafe {
            // Allocate and copy sequences to GPU
            self.cuda.d_seq0 = Some(self.alloc_cuda_seq(
                &seq0.data,
                seq0.length,
                seq0_padding,
                0
            )?);

            self.cuda.d_seq1 = Some(self.alloc_cuda_seq(
                &seq1.data,
                seq1.length,
                seq1_padding,
                0
            )?);

            // Allocate bus memory
            self.host.h_bus_h = Some(
                std::alloc::alloc(
                    std::alloc::Layout::array::<cuda::int2>(bus_size)?
                ) as *mut cuda::int2
            );

            self.cuda.d_bus_h = Some(cuda_malloc(bus_size)?);
            self.cuda.bus_h_size = bus_size as i32;

            // Bind textures (if needed)
            self.bind_textures()?;

            Ok(())
        }
    }

    unsafe fn alloc_cuda_seq(
        &self,
        data: &[u8],
        len: usize,
        padding_len: usize,
        padding_char: u8,
    ) -> Result<*mut u8, CudaAlignerError> {
        let total_len = len + padding_len;
        let out = cuda_malloc(total_len)?;

        // Copy sequence data
        cuda_memcpy(
            out,
            data.as_ptr(),
            len,
            cuda::cudaMemcpyKind_cudaMemcpyHostToDevice,
        )?;

        // Set padding
        if padding_len > 0 {
            cuda_memset(
                out.add(len),
                padding_char as i32,
                padding_len,
            )?;
        }

        Ok(out)
    }

    pub fn align(&mut self) -> Result<AlignmentScore, CudaAlignerError> {
        let start_time = Instant::now();

        // Allocate statistics buffer
        unsafe {
            self.stats_buffer = Some(cuda_malloc(
                (self.params.blocks() as usize) * std::mem::size_of::<cuda::int4>()
            )?);
        }

        // Configure grid and thread dimensions for optimal occupancy
        let (grid, threads) = self.calculate_optimal_grid()?;

        // Initialize block pruning
        let mut active_blocks = self.initialize_block_pruning()?;
        
        // Main alignment loop with block pruning
        for step in 0..self.total_steps() {
            if active_blocks.is_empty() {
                break; // Early termination if no active blocks
            }

            self.process_diagonal(step, &mut active_blocks, grid, threads)?;
        }

        // Collect results
        let score = self.collect_results()?;
        
        // Calculate statistics
        let stats = self.calculate_statistics(start_time.elapsed().as_millis() as f32)?;
        
        Ok(AlignmentScore {
            score: score.0,
            position: score.1,
            statistics: stats,
        })
    }

    fn calculate_optimal_grid(&self) -> Result<(cuda::dim3, cuda::dim3), CudaAlignerError> {
        unsafe {
            let mut max_threads_per_block = 0;
            let mut warp_size = 0;
            
            // Get device properties
            cuda::cudaDeviceGetAttribute(
                &mut max_threads_per_block,
                cuda::cudaDeviceAttr_cudaDevAttrMaxThreadsPerBlock,
                self.params.gpu(),
            );
            
            cuda::cudaDeviceGetAttribute(
                &mut warp_size,
                cuda::cudaDeviceAttr_cudaDevAttrWarpSize,
                self.params.gpu(),
            );

            // Calculate optimal thread count (multiple of warp size)
            let threads_per_block = std::cmp::min(
                THREADS_COUNT as i32,
                (max_threads_per_block / warp_size) * warp_size,
            );

            let grid = cuda::dim3 {
                x: self.params.blocks() as u32,
                y: 1,
                z: 1,
            };

            let threads = cuda::dim3 {
                x: threads_per_block as u32,
                y: 1,
                z: 1,
            };

            Ok((grid, threads))
        }
    }

    fn initialize_block_pruning(&self) -> Result<Vec<i32>, CudaAlignerError> {
        // Initialize with all blocks active
        Ok((0..self.params.blocks()).collect())
    }

    fn process_diagonal(
        &mut self,
        step: i32,
        active_blocks: &mut Vec<i32>,
        grid: cuda::dim3,
        threads: cuda::dim3,
    ) -> Result<(), CudaAlignerError> {
        unsafe {
            let stream = self.stream_processor
                .as_mut()
                .map(|sp| sp.next_stream())
                .unwrap_or(std::ptr::null_mut());

            // Launch kernel with optimizations
            kernel_bindings::launch_kernel_optimized(
                cuda::dim3 { x: active_blocks.len() as u32, ..grid },
                threads,
                self.cuda.d_bus_h.unwrap(),
                self.cuda.d_bus_v_h.unwrap(),
                self.cuda.d_bus_v_e.unwrap(),
                self.cuda.d_bus_v_o.unwrap(),
                self.cuda.d_block_result.unwrap(),
                0,
                self.seq0_len as i32,
                step,
                cuda::int2 { x: 0, y: 0 },
                self.alignment_params.mode as i32,
                self.cuda.d_traceback_matrix.unwrap(),
                self.optimization_config.use_shared_memory,
                stream,
            );

            if self.optimization_config.use_stream_processing {
                cuda::cudaStreamSynchronize(stream);
            } else {
                cuda::cudaDeviceSynchronize();
            }

            // Check for errors
            match cuda::cudaGetLastError() {
                cuda::cudaError_cudaSuccess => Ok(()),
                err => Err(CudaAlignerError::KernelError(
                    format!("Kernel execution failed: {}", err)
                )),
            }?;

            // Update active blocks based on scores
            self.update_active_blocks(active_blocks)?;

            Ok(())
        }
    }

    fn update_active_blocks(&self, active_blocks: &mut Vec<i32>) -> Result<(), CudaAlignerError> {
        // Collect block scores and prune inactive blocks
        let mut new_active_blocks = Vec::new();
        
        unsafe {
            let mut block_results = vec![cuda::int4::default(); active_blocks.len()];
            cuda_memcpy(
                block_results.as_mut_ptr(),
                self.cuda.d_block_result.unwrap(),
                block_results.len() * std::mem::size_of::<cuda::int4>(),
                cuda::cudaMemcpyKind_cudaMemcpyDeviceToHost,
            )?;

            for (i, &block_id) in active_blocks.iter().enumerate() {
                if block_results[i].x > 0 { // Some threshold for activity
                    new_active_blocks.push(block_id);
                }
            }
        }

        *active_blocks = new_active_blocks;
        Ok(())
    }

    fn collect_results(&self) -> Result<(i32, Position), CudaAlignerError> {
        unsafe {
            let mut block_results = vec![cuda::int4::default(); self.params.blocks() as usize];
            cuda_memcpy(
                block_results.as_mut_ptr(),
                self.cuda.d_block_result.unwrap(),
                block_results.len() * std::mem::size_of::<cuda::int4>(),
                cuda::cudaMemcpyKind_cudaMemcpyDeviceToHost,
            )?;

            // Find best score
            let best = block_results.iter()
                .enumerate()
                .max_by_key(|(_, result)| result.x)
                .ok_or_else(|| CudaAlignerError::NoResults)?;

            Ok((best.1.x, Position {
                i: best.1.y,
                j: best.1.z,
            }))
        }
    }

    fn calculate_statistics(&self, execution_time: f32) -> Result<AlignmentStats, CudaAlignerError> {
        unsafe {
            let mut stats = vec![cuda::int4::default(); self.params.blocks() as usize];
            cuda_memcpy(
                stats.as_mut_ptr(),
                self.stats_buffer.unwrap(),
                stats.len() * std::mem::size_of::<cuda::int4>(),
                cuda::cudaMemcpyKind_cudaMemcpyDeviceToHost,
            )?;

            let total_stats = stats.iter().fold(AlignmentStats::default(), |mut acc, &stat| {
                acc.matches += stat.x as u32;
                acc.mismatches += stat.y as u32;
                acc.gaps += stat.z as u32;
                acc
            });

            Ok(AlignmentStats {
                execution_time_ms: execution_time,
                ..total_stats
            })
        }
    }

    fn bind_textures(&self) -> Result<(), CudaAlignerError> {
        unsafe {
            match kernel_bindings::bind_textures(
                self.cuda.d_seq0.unwrap(),
                self.seq0_len,
                self.cuda.d_seq1.unwrap(),
                self.seq1_len,
            ) {
                cuda::cudaError_cudaSuccess => Ok(()),
                err => Err(CudaAlignerError::TextureError(
                    format!("Failed to bind textures: {}", err)
                )),
            }
        }
    }

    fn unbind_textures(&self) -> Result<(), CudaAlignerError> {
        unsafe {
            kernel_bindings::unbind_textures();
            Ok(())
        }
    }

    fn get_memory_usage(&self) -> Result<usize, CudaAlignerError> {
        unsafe {
            let mut free = 0;
            let mut total = 0;
            match cuda::cudaMemGetInfo(&mut free, &mut total) {
                cuda::cudaError_cudaSuccess => Ok(total - free),
                err => Err(CudaAlignerError::MemoryError(
                    format!("Failed to get memory info: {}", err)
                ))
            }
        }
    }

    pub fn unset_sequences(&mut self) -> Result<(), CudaAlignerError> {
        unsafe {
            // Free host memory
            if let Some(ptr) = self.host.h_bus_h.take() {
                std::alloc::dealloc(
                    ptr as *mut u8,
                    std::alloc::Layout::array::<cuda::int2>(self.cuda.bus_h_size as usize)?
                );
            }

            // Free CUDA memory
            if let Some(ptr) = self.cuda.d_seq0.take() {
                cuda_free(ptr)?;
            }
            if let Some(ptr) = self.cuda.d_seq1.take() {
                cuda_free(ptr)?;
            }
            if let Some(ptr) = self.cuda.d_bus_h.take() {
                cuda_free(ptr)?;
            }

            // Unbind textures
            self.unbind_textures()?;

            let used_memory = self.get_memory_usage()?;
            self.stat_deallocated_memory = used_memory - self.stat_initial_used_memory;

            Ok(())
        }
    }

    pub fn with_alignment_mode(mut self, mode: AlignmentMode) -> Self {
        self.alignment_params.mode = mode;
        self
    }

    pub fn align_with_traceback(&mut self) -> Result<AlignmentResult, CudaAlignerError> {
        let score_result = self.align()?;
        
        // Allocate and initialize traceback matrix
        let mut traceback = Traceback::new(&self.seq0, &self.seq1);
        
        // Copy traceback data from GPU
        unsafe {
            cuda_memcpy(
                traceback.matrix.data.as_mut_ptr(),
                self.cuda.d_traceback_matrix.unwrap(),
                traceback.matrix.data.len(),
                cuda::cudaMemcpyKind_cudaMemcpyDeviceToHost,
            )?;
        }

        // Reconstruct alignment
        traceback.reconstruct_alignment(
            score_result.position,
            Position { 
                i: self.seq0_len as i32, 
                j: self.seq1_len as i32 
            },
            self.alignment_params.mode,
        )
    }

    pub fn with_optimizations(mut self, config: OptimizationConfig) -> Self {
        self.optimization_config = config;
        if config.use_stream_processing {
            self.stream_processor = StreamProcessor::new(4).ok();
        }
        self
    }
}

// Implement proper cleanup
impl Drop for CudaAligner {
    fn drop(&mut self) {
        unsafe {
            // Clean up sequences
            let _ = self.unset_sequences();

            // Clean up host memory
            if let Some(ptr) = self.host.h_extra_h.take() {
                std::alloc::dealloc(
                    ptr as *mut u8,
                    std::alloc::Layout::array::<cuda::int2>(THREADS_COUNT as usize)
                        .expect("Invalid layout")
                );
            }
            // ... (similar cleanup for other host structures)

            // Clean up CUDA memory
            if let Some(ptr) = self.cuda.d_extra_h.take() {
                let _ = cuda_free(ptr);
            }
            // ... (similar cleanup for other CUDA structures)

            // Reset device
            let _ = cuda::cudaDeviceReset();
        }
    }
}

#[derive(Debug)]
pub struct AlignmentResult {
    pub score: i32,
    // Add other fields as needed
}