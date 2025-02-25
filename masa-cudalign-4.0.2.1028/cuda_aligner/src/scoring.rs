use crate::CudaAlignerError;
use cuda_runtime_sys as cuda;

#[derive(Debug, Clone)]
pub struct AlignmentScore {
    pub score: i32,
    pub position: Position,
    pub statistics: AlignmentStats,
}

#[derive(Debug, Clone, Copy)]
pub struct Position {
    pub i: i32,
    pub j: i32,
}

#[derive(Debug, Clone, Default)]
pub struct AlignmentStats {
    pub matches: u32,
    pub mismatches: u32,
    pub gaps: u32,
    pub execution_time_ms: f32,
}

pub(crate) struct BlockScores {
    scores: Vec<i32>,
    positions: Vec<Position>,
}

impl BlockScores {
    pub fn new(capacity: usize) -> Self {
        Self {
            scores: Vec::with_capacity(capacity),
            positions: Vec::with_capacity(capacity),
        }
    }

    pub fn add_block_score(&mut self, score: i32, pos: Position) {
        self.scores.push(score);
        self.positions.push(pos);
    }

    pub fn get_best_score(&self) -> Option<(i32, Position)> {
        self.scores.iter()
            .zip(self.positions.iter())
            .max_by_key(|(&score, _)| score)
            .map(|(&score, &pos)| (score, pos))
    }
}
