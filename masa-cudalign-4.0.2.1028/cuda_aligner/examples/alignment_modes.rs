use cuda_aligner::{
    CudaAligner, 
    CudaAlignerParameters, 
    Sequence, 
    AlignmentMode,
    AlignmentParams,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize aligner with global alignment
    let params = CudaAlignerParameters::new()
        .with_blocks(256)?;
    
    let mut aligner = CudaAligner::new(params)?
        .with_alignment_mode(AlignmentMode::Global);
    
    // Initialize sequences
    let seq1 = Sequence::new(b"ACGTACGT");
    let seq2 = Sequence::new(b"ACGTAGCT");

    // Perform alignment with traceback
    let result = aligner.align_with_traceback()?;

    // Print alignment
    println!("Score: {}", result.score);
    println!("Aligned sequence 1: {}", String::from_utf8_lossy(&result.aligned_seq1));
    println!("Aligned sequence 2: {}", String::from_utf8_lossy(&result.aligned_seq2));
    println!("Operations: {:?}", result.operations);

    Ok(())
} 