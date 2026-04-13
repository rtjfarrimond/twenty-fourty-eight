use game_engine::{all_afterstates, Board, Direction, MoveTables};
use std::sync::Arc;

use crate::Agent;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::_pext_u64;

#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn pext(source: u64, mask: u64) -> u64 {
    unsafe { _pext_u64(source, mask) }
}

const TILE_VALUES: usize = 16;
const DIRECTIONS: [Direction; 4] = [
    Direction::Left,
    Direction::Right,
    Direction::Up,
    Direction::Down,
];

/// An n-tuple network agent loaded from a binary model file.
/// Uses the same isomorphic evaluation as the training code.
pub struct NTupleAgent {
    masks: Vec<u64>,
    weights: Vec<f32>,
    table_size: usize,
    tables: Arc<MoveTables>,
}

impl NTupleAgent {
    /// Loads a model from a binary file produced by the training binary.
    pub fn load(path: &str, tables: Arc<MoveTables>) -> std::io::Result<Self> {
        use std::io::Read;
        let mut file = std::io::BufReader::new(std::fs::File::open(path)?);

        let mut buf4 = [0u8; 4];
        let mut buf8 = [0u8; 8];

        file.read_exact(&mut buf4)?;
        let num_masks = u32::from_le_bytes(buf4) as usize;

        let mut masks = Vec::with_capacity(num_masks);
        for _ in 0..num_masks {
            file.read_exact(&mut buf8)?;
            masks.push(u64::from_le_bytes(buf8));
        }

        let table_size = TILE_VALUES.pow(6);
        let total_weights = num_masks * table_size;
        let mut weights = Vec::with_capacity(total_weights);
        for _ in 0..total_weights {
            file.read_exact(&mut buf4)?;
            weights.push(f32::from_le_bytes(buf4));
        }

        Ok(Self {
            masks,
            weights,
            table_size,
            tables,
        })
    }

    #[inline(always)]
    fn flip(raw: u64) -> u64 {
        let buf = (raw ^ raw.rotate_left(16)) & 0x0000ffff0000ffff;
        raw ^ (buf | buf.rotate_right(16))
    }

    #[inline(always)]
    fn transpose(raw: u64) -> u64 {
        let a = raw;
        let t = (a ^ (a >> 12)) & 0x0000f0f00000f0f0;
        let a = a ^ t ^ (t << 12);
        let t = (a ^ (a >> 24)) & 0x00000000ff00ff00;
        a ^ t ^ (t << 24)
    }

    #[inline(always)]
    fn evaluate_orientation(&self, raw: u64) -> f32 {
        let weights_ptr = self.weights.as_ptr();
        let mut total = 0.0f32;
        for (pattern_index, &mask) in self.masks.iter().enumerate() {
            let index = pext(raw, mask) as usize;
            let offset = pattern_index * self.table_size + index;
            unsafe {
                total += *weights_ptr.add(offset);
            }
        }
        total
    }

    fn evaluate_board(&self, board: &Board) -> f32 {
        let raw = board.raw();
        let r0 = raw;
        let r1 = Self::flip(r0);
        let r2 = Self::transpose(r1);
        let r3 = Self::flip(r2);
        let r4 = Self::transpose(r3);
        let r5 = Self::flip(r4);
        let r6 = Self::transpose(r5);
        let r7 = Self::flip(r6);

        self.evaluate_orientation(r0)
            + self.evaluate_orientation(r1)
            + self.evaluate_orientation(r2)
            + self.evaluate_orientation(r3)
            + self.evaluate_orientation(r4)
            + self.evaluate_orientation(r5)
            + self.evaluate_orientation(r6)
            + self.evaluate_orientation(r7)
    }
}

impl Agent for NTupleAgent {
    fn best_move(&self, board: &Board) -> Direction {
        let afterstates = all_afterstates(board, &self.tables);
        let mut best_direction = Direction::Down;
        let mut best_total = f32::NEG_INFINITY;

        for (index, &(afterstate, reward, changed)) in afterstates.iter().enumerate() {
            if !changed {
                continue;
            }
            let total = reward as f32 + self.evaluate_board(&afterstate);
            if total > best_total {
                best_total = total;
                best_direction = DIRECTIONS[index];
            }
        }

        best_direction
    }

    fn evaluate(&self, board: &Board) -> f64 {
        self.evaluate_board(board) as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ntuple_agent_implements_agent_trait() {
        // This test just verifies the type system works.
        // A real test would need a saved model file.
        fn accepts_agent(_agent: &dyn Agent) {}
        // We can't construct one without a file, but we can verify the trait bound.
        let _: fn(&NTupleAgent) = accepts_agent;
    }
}
