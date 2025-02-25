#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AlignmentMode {
    Global,      // Needleman-Wunsch
    Local,       // Smith-Waterman
    SemiGlobal,  // Free end-gaps
}

#[derive(Debug, Clone, Copy)]
pub struct AlignmentParams {
    pub mode: AlignmentMode,
    pub gap_open: i32,
    pub gap_extend: i32,
    pub match_score: i32,
    pub mismatch_score: i32,
}

impl Default for AlignmentParams {
    fn default() -> Self {
        Self {
            mode: AlignmentMode::Global,
            gap_open: DNA_GAP_OPEN,
            gap_extend: DNA_GAP_EXT,
            match_score: DNA_MATCH,
            mismatch_score: DNA_MISMATCH,
        }
    }
}
