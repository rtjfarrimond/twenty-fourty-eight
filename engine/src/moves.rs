/// Extracts the 4 tile exponents from a packed 16-bit row.
fn unpack_row(packed_row: u16) -> [u8; 4] {
    [
        (packed_row & 0xF) as u8,
        ((packed_row >> 4) & 0xF) as u8,
        ((packed_row >> 8) & 0xF) as u8,
        ((packed_row >> 12) & 0xF) as u8,
    ]
}

/// Packs 4 tile exponents into a 16-bit row.
fn pack_row(tiles: &[u8; 4]) -> u16 {
    (tiles[0] as u16)
        | ((tiles[1] as u16) << 4)
        | ((tiles[2] as u16) << 8)
        | ((tiles[3] as u16) << 12)
}

/// Slides a single row to the left, merging adjacent equal tiles.
/// Input and output are packed 16-bit rows (4 nibbles, low = leftmost tile).
/// Returns (result_row, score) where score is the sum of merged tile values.
pub fn slide_row_left(packed_row: u16) -> (u16, u32) {
    let tiles = unpack_row(packed_row);
    let mut score: u32 = 0;

    // Compact: remove gaps by collecting non-zero tiles
    let mut compacted = [0u8; 4];
    let mut write_pos = 0;
    for &tile in &tiles {
        if tile != 0 {
            compacted[write_pos] = tile;
            write_pos += 1;
        }
    }

    // Merge adjacent equal tiles left-to-right
    let mut result = [0u8; 4];
    let mut read_pos = 0;
    let mut out_pos = 0;
    while read_pos < 4 {
        if compacted[read_pos] == 0 {
            break;
        }
        if read_pos + 1 < 4 && compacted[read_pos] == compacted[read_pos + 1] {
            let merged_exponent = compacted[read_pos] + 1;
            result[out_pos] = merged_exponent;
            score += 1 << merged_exponent;
            read_pos += 2;
        } else {
            result[out_pos] = compacted[read_pos];
            read_pos += 1;
        }
        out_pos += 1;
    }

    (pack_row(&result), score)
}

/// Reverses the tile order within a packed 16-bit row.
/// Used to derive slide-right from slide-left.
fn reverse_row(packed_row: u16) -> u16 {
    let tiles = unpack_row(packed_row);
    pack_row(&[tiles[3], tiles[2], tiles[1], tiles[0]])
}

/// Precomputed lookup tables for row moves and scores.
pub struct MoveTables {
    /// Result of sliding each possible row left.
    pub left_result: Vec<u16>,
    /// Score gained from sliding each possible row left.
    pub left_score: Vec<u32>,
    /// Result of sliding each possible row right.
    pub right_result: Vec<u16>,
    /// Score gained from sliding each possible row right.
    pub right_score: Vec<u32>,
}

impl MoveTables {
    pub fn new() -> Self {
        let mut left_result = vec![0u16; 65536];
        let mut left_score = vec![0u32; 65536];
        let mut right_result = vec![0u16; 65536];
        let mut right_score = vec![0u32; 65536];

        for packed_row in 0..65536u32 {
            let row = packed_row as u16;
            let (result, score) = slide_row_left(row);
            left_result[packed_row as usize] = result;
            left_score[packed_row as usize] = score;

            let reversed = reverse_row(row);
            let (right_slid, right_scr) = slide_row_left(reversed);
            right_result[packed_row as usize] = reverse_row(right_slid);
            right_score[packed_row as usize] = right_scr;
        }

        Self {
            left_result,
            left_score,
            right_result,
            right_score,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slide_empty_row_is_noop() {
        let (result, score) = slide_row_left(0);
        assert_eq!(result, 0);
        assert_eq!(score, 0);
    }

    #[test]
    fn slide_single_tile_moves_to_left() {
        // [0, 0, 1, 0] → [1, 0, 0, 0]
        let input: u16 = 1 << 8; // exponent 1 at position 2
        let (result, score) = slide_row_left(input);
        assert_eq!(result, 1); // exponent 1 at position 0
        assert_eq!(score, 0);
    }

    #[test]
    fn slide_merges_equal_adjacent_tiles() {
        // [1, 1, 0, 0] → [2, 0, 0, 0] with score = 4 (two 2-tiles merge to 4)
        let input: u16 = 1 | (1 << 4);
        let (result, score) = slide_row_left(input);
        assert_eq!(result, 2); // exponent 2 at position 0
        assert_eq!(score, 4); // merged tile value = 2^2 = 4
    }

    #[test]
    fn slide_does_not_merge_twice_in_one_move() {
        // [1, 1, 1, 1] → [2, 2, 0, 0] not [3, 0, 0, 0]
        let input: u16 = 1 | (1 << 4) | (1 << 8) | (1 << 12);
        let (result, score) = slide_row_left(input);
        let expected: u16 = 2 | (2 << 4);
        assert_eq!(result, expected);
        assert_eq!(score, 8); // two merges of 2→4, each scores 4
    }

    #[test]
    fn slide_merges_nearest_pair_first() {
        // [1, 1, 1, 0] → [2, 1, 0, 0]
        let input: u16 = 1 | (1 << 4) | (1 << 8);
        let (result, score) = slide_row_left(input);
        let expected: u16 = 2 | (1 << 4);
        assert_eq!(result, expected);
        assert_eq!(score, 4);
    }

    #[test]
    fn slide_does_not_merge_unequal_tiles() {
        // [1, 2, 0, 0] → [1, 2, 0, 0]
        let input: u16 = 1 | (2 << 4);
        let (result, score) = slide_row_left(input);
        assert_eq!(result, input);
        assert_eq!(score, 0);
    }

    #[test]
    fn slide_compacts_gaps() {
        // [1, 0, 2, 0] → [1, 2, 0, 0]
        let input: u16 = 1 | (2 << 8);
        let (result, score) = slide_row_left(input);
        let expected: u16 = 1 | (2 << 4);
        assert_eq!(result, expected);
        assert_eq!(score, 0);
    }

    #[test]
    fn slide_merges_after_compacting_gap() {
        // [1, 0, 1, 0] → [2, 0, 0, 0]
        let input: u16 = 1 | (1 << 8);
        let (result, score) = slide_row_left(input);
        assert_eq!(result, 2);
        assert_eq!(score, 4);
    }

    #[test]
    fn move_tables_left_matches_slide_row_left() {
        let tables = MoveTables::new();
        // Spot check: [1, 1, 0, 0] → [2, 0, 0, 0]
        let input: u16 = 1 | (1 << 4);
        assert_eq!(tables.left_result[input as usize], 2);
        assert_eq!(tables.left_score[input as usize], 4);
    }

    #[test]
    fn move_tables_right_mirrors_left() {
        let tables = MoveTables::new();
        // [0, 0, 1, 1] sliding right → [0, 0, 0, 2]
        let input: u16 = (1 << 8) | (1 << 12);
        let expected: u16 = 2 << 12;
        assert_eq!(tables.right_result[input as usize], expected);
        assert_eq!(tables.right_score[input as usize], 4);
    }
}
