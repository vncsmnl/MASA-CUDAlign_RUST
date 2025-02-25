#[derive(Debug, Clone)]
pub struct PerformanceConfig {
    pub block_pruning_threshold: i32,
    pub early_termination: bool,
    pub use_texture_memory: bool,
    pub shared_memory_per_block: usize,
}

impl Default for PerformanceConfig {
    fn default() -> Self {
        Self {
            block_pruning_threshold: 0,
            early_termination: true,
            use_texture_memory: true,
            shared_memory_per_block: 48 * 1024, // 48KB default
        }
    }
}
