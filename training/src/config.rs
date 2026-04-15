//! Shared CLI-adjacent helpers: pattern presets and algorithm validation.
//! Used by both the `training` binary and the `benchmark` binary so the two
//! stay in lockstep on what's valid and which patterns each preset refers to.

/// Standard 4-pattern 6-tuple configuration from the literature.
pub fn patterns_4x6() -> Vec<Vec<(usize, usize)>> {
    vec![
        vec![(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1)],
        vec![(1, 0), (1, 1), (1, 2), (1, 3), (2, 0), (2, 1)],
        vec![(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)],
        vec![(1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)],
    ]
}

/// Stronger 8-pattern 6-tuple configuration (adds 4 more L/rectangle shapes).
/// From RESEARCH.md — flat indices converted to (row, col).
pub fn patterns_8x6() -> Vec<Vec<(usize, usize)>> {
    let mut patterns = patterns_4x6();
    patterns.extend(vec![
        vec![(0, 0), (0, 1), (1, 1), (1, 2), (1, 3), (2, 2)],
        vec![(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (2, 1)],
        vec![(0, 0), (0, 1), (0, 2), (1, 1), (2, 1), (2, 2)],
        vec![(0, 0), (0, 1), (1, 1), (1, 2), (2, 1), (3, 1)],
    ]);
    patterns
}

pub fn select_patterns(preset: &str) -> Vec<Vec<(usize, usize)>> {
    match preset {
        "4x6" => patterns_4x6(),
        "8x6" => patterns_8x6(),
        other => panic!("Unknown pattern preset: {other}. Expected 4x6 or 8x6."),
    }
}

/// Rejects a CLI invocation that asks for both optimistic and random
/// weight initialization. They're orthogonal concepts and combining them
/// is almost certainly a user error.
pub fn validate_init_mode(optimistic_init: f32, random_init_amplitude: f32) {
    if optimistic_init != 0.0 && random_init_amplitude != 0.0 {
        panic!(
            "--optimistic-init and --random-init-amplitude are mutually exclusive \
             (got optimistic={optimistic_init}, random_amp={random_init_amplitude}). \
             Pick one or leave both at 0."
        );
    }
}

/// Validates the (algorithm, threads) combination at CLI entry.
/// Panics with a user-facing message on invalid combinations.
pub fn validate_algorithm(algorithm: &str, threads: u32) {
    match algorithm {
        "serial" => {
            if threads != 1 {
                panic!(
                    "serial algorithm requires --threads 1 (got {threads}). \
                     Use --algorithm hogwild for multi-threaded training."
                );
            }
        }
        "hogwild" => {
            if threads == 0 {
                panic!("hogwild algorithm requires --threads >= 1 (got 0).");
            }
        }
        other => panic!(
            "Unknown algorithm: {other}. Expected \"serial\" or \"hogwild\"."
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn patterns_4x6_has_four_base_patterns() {
        assert_eq!(patterns_4x6().len(), 4);
    }

    #[test]
    fn patterns_8x6_has_eight_base_patterns() {
        assert_eq!(patterns_8x6().len(), 8);
    }

    #[test]
    fn patterns_8x6_is_superset_of_4x6() {
        let four = patterns_4x6();
        let eight = patterns_8x6();
        for pattern in &four {
            assert!(eight.contains(pattern));
        }
    }

    #[test]
    fn select_patterns_routes_by_preset() {
        assert_eq!(select_patterns("4x6").len(), 4);
        assert_eq!(select_patterns("8x6").len(), 8);
    }

    #[test]
    #[should_panic(expected = "Unknown pattern preset: garbage")]
    fn select_patterns_rejects_unknown() {
        select_patterns("garbage");
    }

    #[test]
    fn serial_with_single_thread_is_valid() {
        validate_algorithm("serial", 1);
    }

    #[test]
    #[should_panic(expected = "serial algorithm requires --threads 1")]
    fn serial_with_multiple_threads_rejected() {
        validate_algorithm("serial", 4);
    }

    #[test]
    #[should_panic(expected = "serial algorithm requires --threads 1")]
    fn serial_with_zero_threads_rejected() {
        validate_algorithm("serial", 0);
    }

    #[test]
    fn hogwild_with_positive_thread_count_is_valid() {
        validate_algorithm("hogwild", 4);
        validate_algorithm("hogwild", 1);
    }

    #[test]
    #[should_panic(expected = "hogwild algorithm requires --threads >= 1")]
    fn hogwild_with_zero_threads_rejected() {
        validate_algorithm("hogwild", 0);
    }

    #[test]
    #[should_panic(expected = "Unknown algorithm: mystery")]
    fn unknown_algorithm_rejected() {
        validate_algorithm("mystery", 1);
    }

    #[test]
    fn init_mode_both_zero_is_valid() {
        validate_init_mode(0.0, 0.0);
    }

    #[test]
    fn init_mode_optimistic_only_is_valid() {
        validate_init_mode(100.0, 0.0);
    }

    #[test]
    fn init_mode_random_only_is_valid() {
        validate_init_mode(0.0, 0.01);
    }

    #[test]
    #[should_panic(expected = "mutually exclusive")]
    fn init_mode_both_nonzero_rejected() {
        validate_init_mode(100.0, 0.01);
    }
}
