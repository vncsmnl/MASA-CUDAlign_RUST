use cuda_aligner::{CudaAligner, CudaAlignerParameters, Sequence};
use env_logger;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();

    let params = CudaAlignerParameters::new()
        .with_blocks(256)?;
    
    let mut aligner = CudaAligner::new(params)?;
    aligner.initialize()?;

    let seq0 = Sequence::new(b"ACGTACGT");
    let seq1 = Sequence::new(b"ACGTAGCT");

    aligner.set_sequences(&seq0, &seq1)?;

    println!("Sequences loaded successfully!");

    aligner.unset_sequences()?;

    Ok(())
}