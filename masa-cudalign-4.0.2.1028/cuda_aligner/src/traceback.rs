use crate::{CudaAlignerError, Position};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TracebackOperation {
    Match,
    Mismatch,
    Insertion,
    Deletion,
}

#[derive(Debug, Clone)]
pub struct AlignmentResult {
    pub score: i32,
    pub aligned_seq1: Vec<u8>,
    pub aligned_seq2: Vec<u8>,
    pub operations: Vec<TracebackOperation>,
    pub start_position: Position,
    pub end_position: Position,
}

pub struct TracebackMatrix {
    data: Vec<u8>,
    rows: usize,
    cols: usize,
}

impl TracebackMatrix {
    pub fn new(rows: usize, cols: usize) -> Self {
        Self {
            data: vec![0; rows * cols],
            rows,
            cols,
        }
    }

    pub fn set(&mut self, row: usize, col: usize, value: u8) {
        self.data[row * self.cols + col] = value;
    }

    pub fn get(&self, row: usize, col: usize) -> u8 {
        self.data[row * self.cols + col]
    }
}

pub struct Traceback {
    matrix: TracebackMatrix,
    seq1: Vec<u8>,
    seq2: Vec<u8>,
}

impl Traceback {
    pub fn new(seq1: &[u8], seq2: &[u8]) -> Self {
        Self {
            matrix: TracebackMatrix::new(seq1.len() + 1, seq2.len() + 1),
            seq1: seq1.to_vec(),
            seq2: seq2.to_vec(),
        }
    }

    pub fn reconstruct_alignment(
        &self,
        start: Position,
        end: Position,
        mode: AlignmentMode,
    ) -> Result<AlignmentResult, CudaAlignerError> {
        let mut aligned_seq1 = Vec::new();
        let mut aligned_seq2 = Vec::new();
        let mut operations = Vec::new();

        let mut current = end;
        let mut score = 0;

        while current != start {
            let op = self.matrix.get(current.i as usize, current.j as usize);
            match op {
                0 => { // Match/Mismatch
                    aligned_seq1.push(self.seq1[current.i as usize - 1]);
                    aligned_seq2.push(self.seq2[current.j as usize - 1]);
                    if self.seq1[current.i as usize - 1] == self.seq2[current.j as usize - 1] {
                        operations.push(TracebackOperation::Match);
                        score += DNA_MATCH;
                    } else {
                        operations.push(TracebackOperation::Mismatch);
                        score -= DNA_MISMATCH;
                    }
                    current.i -= 1;
                    current.j -= 1;
                },
                1 => { // Deletion
                    aligned_seq1.push(self.seq1[current.i as usize - 1]);
                    aligned_seq2.push(b'-');
                    operations.push(TracebackOperation::Deletion);
                    score -= DNA_GAP_FIRST;
                    current.i -= 1;
                },
                2 => { // Insertion
                    aligned_seq1.push(b'-');
                    aligned_seq2.push(self.seq2[current.j as usize - 1]);
                    operations.push(TracebackOperation::Insertion);
                    score -= DNA_GAP_FIRST;
                    current.j -= 1;
                },
                _ => return Err(CudaAlignerError::TracebackError(
                    "Invalid traceback operation".to_string()
                )),
            }
        }

        // Reverse the sequences and operations since we built them backwards
        aligned_seq1.reverse();
        aligned_seq2.reverse();
        operations.reverse();

        Ok(AlignmentResult {
            score,
            aligned_seq1,
            aligned_seq2,
            operations,
            start_position: start,
            end_position: end,
        })
    }
}
