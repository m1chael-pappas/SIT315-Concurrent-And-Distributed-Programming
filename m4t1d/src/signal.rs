//! Actuated traffic signals.
//!
//! Each intersection alternates between two phases: north-south green (phase 0)
//! and east-west green (phase 1). A detector covers the last few cells of every
//! approach. The light holds green for at least the minimum time, keeps it
//! while the green approaches still have cars on their detectors, and switches
//! once the red approaches are waiting and either the green side has gone quiet
//! or the maximum green time is up. With nobody waiting on red it rests in green.
//! All of this reads only the intersection's own approaches.

/// Phase 0 gives green to the north and south approaches, phase 1 to east and west.
pub fn is_green(phase: u8, approach: usize) -> bool {
    approach % 2 == phase as usize
}

/// Signal state for the next tick, from the current phase, how long it has
/// lasted, and whether the green and red approaches have cars on their detectors.
pub fn next_signal(
    phase: u8,
    elapsed: u16,
    green_waiting: bool,
    red_waiting: bool,
    min_green: u32,
    max_green: u32,
) -> (u8, u16) {
    let elapsed = elapsed.saturating_add(1);
    let held = u32::from(elapsed);
    let switch = held >= min_green && red_waiting && (held >= max_green || !green_waiting);
    if switch { (phase ^ 1, 0) } else { (phase, elapsed) }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MIN: u32 = 10;
    const MAX: u32 = 45;

    #[test]
    fn phases_pair_opposite_approaches() {
        assert!(is_green(0, 0) && is_green(0, 2));
        assert!(!is_green(0, 1) && !is_green(0, 3));
        assert!(is_green(1, 1) && is_green(1, 3));
    }

    #[test]
    fn holds_until_minimum_green() {
        assert_eq!(next_signal(0, 5, false, true, MIN, MAX), (0, 6));
    }

    #[test]
    fn gaps_out_when_green_side_is_empty() {
        assert_eq!(next_signal(0, 9, false, true, MIN, MAX), (1, 0));
    }

    #[test]
    fn extends_while_green_side_is_busy() {
        assert_eq!(next_signal(0, 20, true, true, MIN, MAX), (0, 21));
    }

    #[test]
    fn maxes_out_under_demand() {
        assert_eq!(next_signal(1, 44, true, true, MIN, MAX), (0, 0));
    }

    #[test]
    fn rests_in_green_when_nobody_waits() {
        assert_eq!(next_signal(0, 100, false, false, MIN, MAX), (0, 101));
        assert_eq!(next_signal(0, u16::MAX, true, false, MIN, MAX), (0, u16::MAX));
    }
}
