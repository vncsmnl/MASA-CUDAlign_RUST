use cuda_aligner::{
    MultiGpuManager,
    Sequence,
    AlignmentMode,
    OptimizationConfig,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize multi-GPU manager
    let manager = MultiGpuManager::new()?;

    // Create test sequences
    let sequences = vec![
        (Sequence::new(b"ACGTACGT"), Sequence::new(b"ACGTAGCT")),
        (Sequence::new(b"GGTTAACC"), Sequence::new(b"GGTTAACT")),
        // Add more sequence pairs...
    ];

    // Configure optimizations
    let config = OptimizationConfig {
        use_shared_memory: true,
        use_texture_memory: true,
        use_pinned_memory: true,
        use_stream_processing: true,
        block_pruning_threshold: 0.1,
    };

    // Perform parallel alignment
    let results = manager.align_parallel(sequences, AlignmentMode::Global)?;

    // Process results
    for (i, result) in results.iter().enumerate() {
        println!("Alignment {}: Score = {}", i, result.score);
        println!("Sequence 1: {}", String::from_utf8_lossy(&result.aligned_seq1));
        println!("Sequence 2: {}", String::from_utf8_lossy(&result.aligned_seq2));
        println!();
    }

    Ok(())
} 