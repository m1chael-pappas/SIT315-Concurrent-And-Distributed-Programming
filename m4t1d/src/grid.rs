//! Grid geometry: intersections, approach links and turns.
//!
//! Intersections are numbered row-major, n = r * cols + c, and that number is
//! the traffic light id in the sensor CSV. Every intersection owns its four
//! incoming approaches, link id 4n + d, where d is the side the cars arrive
//! from: north 0, east 1, south 2, west 3. Rows grow southwards.
//!
//! Traffic keeps left, as in Australia. A car arriving from side d leaves
//! through port (d + 2) % 4 going straight, (d + 1) % 4 turning left and
//! (d + 3) % 4 turning right. Right turns cross the oncoming lane.

/// North side: cars arriving from the intersection above, or leaving towards it.
pub const NORTH: usize = 0;
/// East side.
pub const EAST: usize = 1;
/// South side.
pub const SOUTH: usize = 2;
/// West side.
pub const WEST: usize = 3;

/// The side facing `d`.
pub fn opposite(d: usize) -> usize {
    (d + 2) % 4
}

/// Link id of intersection `node`'s approach from side `d`.
pub fn link_id(node: usize, d: usize) -> usize {
    4 * node + d
}

/// The intersection next to (r, c) on side `d`, or None at the city edge.
pub fn neighbour(rows: u32, cols: u32, r: u32, c: u32, d: usize) -> Option<(u32, u32)> {
    match d {
        NORTH if r > 0 => Some((r - 1, c)),
        EAST if c + 1 < cols => Some((r, c + 1)),
        SOUTH if r + 1 < rows => Some((r + 1, c)),
        WEST if c > 0 => Some((r, c - 1)),
        _ => None,
    }
}

/// Neighbour of intersection `node` on side `d` as an intersection number.
pub fn neighbour_node(rows: u32, cols: u32, node: usize, d: usize) -> Option<usize> {
    let r = (node / cols as usize) as u32;
    let c = (node % cols as usize) as u32;
    neighbour(rows, cols, r, c, d).map(|(nr, nc)| (nr * cols + nc) as usize)
}

/// Port a car leaves through, given the side it arrived from and its turn code (0 straight, 1 left, 2 right).
pub fn exit_port(approach: usize, turn: u8) -> usize {
    match turn {
        0 => (approach + 2) % 4,
        1 => (approach + 1) % 4,
        _ => (approach + 3) % 4,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keep_left_turns() {
        assert_eq!(exit_port(NORTH, 0), SOUTH);
        assert_eq!(exit_port(NORTH, 1), EAST);
        assert_eq!(exit_port(NORTH, 2), WEST);
        assert_eq!(exit_port(EAST, 0), WEST);
        assert_eq!(exit_port(EAST, 1), SOUTH);
        assert_eq!(exit_port(EAST, 2), NORTH);
        assert_eq!(exit_port(SOUTH, 1), WEST);
        assert_eq!(exit_port(WEST, 1), NORTH);
    }

    #[test]
    fn neighbours_stop_at_edges() {
        assert_eq!(neighbour(3, 4, 0, 0, NORTH), None);
        assert_eq!(neighbour(3, 4, 0, 0, WEST), None);
        assert_eq!(neighbour(3, 4, 0, 0, EAST), Some((0, 1)));
        assert_eq!(neighbour(3, 4, 0, 0, SOUTH), Some((1, 0)));
        assert_eq!(neighbour(3, 4, 2, 3, SOUTH), None);
        assert_eq!(neighbour(3, 4, 2, 3, EAST), None);
        assert_eq!(neighbour_node(3, 4, 5, NORTH), Some(1));
        assert_eq!(neighbour_node(3, 4, 5, WEST), Some(4));
    }

    #[test]
    fn single_intersection_has_no_neighbours() {
        for d in 0..4 {
            assert_eq!(neighbour(1, 1, 0, 0, d), None);
        }
    }

    #[test]
    fn opposite_is_an_involution() {
        for d in 0..4 {
            assert_eq!(opposite(opposite(d)), d);
            assert_ne!(opposite(d), d);
        }
    }
}
