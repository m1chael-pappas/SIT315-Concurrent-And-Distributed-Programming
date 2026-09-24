//! Philox-4x32-10, the counter-based random number generator from Salmon et al.,
//! "Parallel Random Numbers: As Easy as 1, 2, 3" (SC'11, Random123).
//!
//! Every random decision in the simulation is a pure function of the seed, the
//! entity it belongs to, a time or hop number and a stream tag. There is no
//! generator state to share or advance, so the numbers a car sees do not depend
//! on which thread, backend or partition processes it. `gpu/kernels.cu` holds
//! the CUDA copy of this function, and both are tested against the Random123
//! known-answer vectors.

/// Multiplier of the first counter word in each round.
pub const M0: u32 = 0xD251_1F53;
/// Multiplier of the third counter word in each round.
pub const M1: u32 = 0xCD9E_8D57;
/// Weyl increment of the first key word between rounds.
pub const W0: u32 = 0x9E37_79B9;
/// Weyl increment of the second key word between rounds.
pub const W1: u32 = 0xBB67_AE85;
/// Rounds, the 10 in Philox-4x32-10.
pub const ROUNDS: u32 = 10;

/// Stream tag for the random braking step of the Nagel-Schreckenberg rules.
pub const STREAM_BRAKE: u32 = 1;
/// Stream tag for the turn and trip-end decision a car makes on entering a link.
pub const STREAM_ROUTE: u32 = 2;
/// Stream tag for whether a link spawns a car in a tick.
pub const STREAM_SPAWN: u32 = 3;

/// Random123's published known-answer vectors for Philox-4x32-10: counter, key, expected output.
pub const KNOWN_ANSWERS: [([u32; 4], [u32; 2], [u32; 4]); 3] = [
    ([0, 0, 0, 0], [0, 0], [0x6627_e8d5, 0xe169_c58d, 0xbc57_ac4c, 0x9b00_dbd8]),
    ([u32::MAX; 4], [u32::MAX; 2], [0x408f_276d, 0x41c8_3b0e, 0xa20b_c7c6, 0x6d54_51fd]),
    (
        [0x243f_6a88, 0x85a3_08d3, 0x1319_8a2e, 0x0370_7344],
        [0xa409_3822, 0x299f_31d0],
        [0xd16c_fe09, 0x94fd_cceb, 0x5001_e420, 0x2412_6ea1],
    ),
];

/// The counter and key of each known-answer vector, in order.
pub fn known_answer_inputs() -> [([u32; 4], [u32; 2]); 3] {
    KNOWN_ANSWERS.map(|(ctr, key, _)| (ctr, key))
}

/// How many of `outputs`, computed from `known_answer_inputs` in order, equal the published answers.
pub fn known_answers_matched(outputs: &[[u32; 4]]) -> usize {
    KNOWN_ANSWERS.iter().zip(outputs).filter(|((_, _, want), got)| want == *got).count()
}

/// A Philox key built from a 64-bit seed, low word first.
pub fn key_from_seed(seed: u64) -> [u32; 2] {
    [seed as u32, (seed >> 32) as u32]
}

fn mulhilo(a: u32, b: u32) -> (u32, u32) {
    let product = u64::from(a) * u64::from(b);
    ((product >> 32) as u32, product as u32)
}

fn round(ctr: [u32; 4], key: [u32; 2]) -> [u32; 4] {
    let (hi0, lo0) = mulhilo(M0, ctr[0]);
    let (hi1, lo1) = mulhilo(M1, ctr[2]);
    [hi1 ^ ctr[1] ^ key[0], lo1, hi0 ^ ctr[3] ^ key[1], lo0]
}

/// Philox-4x32 with 10 rounds: four 32-bit outputs from a 128-bit counter and a 64-bit key.
pub fn philox4x32_10(ctr: [u32; 4], key: [u32; 2]) -> [u32; 4] {
    let mut ctr = ctr;
    let mut key = key;
    for r in 0..ROUNDS {
        if r > 0 {
            key = [key[0].wrapping_add(W0), key[1].wrapping_add(W1)];
        }
        ctr = round(ctr, key);
    }
    ctr
}

/// The braking draw for a car at a tick, compared against the braking threshold.
pub fn draw_brake(key: [u32; 2], spawn_link: u32, spawn_tick: u32, tick: u32) -> u32 {
    philox4x32_10([spawn_link, spawn_tick, tick, STREAM_BRAKE], key)[0]
}

/// The route draws for a car entering its `hop`-th link: word 0 picks the turn, word 1 the trip end.
pub fn draw_route(key: [u32; 2], spawn_link: u32, spawn_tick: u32, hop: u32) -> [u32; 4] {
    philox4x32_10([spawn_link, spawn_tick, hop, STREAM_ROUTE], key)
}

/// The spawn draw for a link at a tick, compared against the link's spawn threshold.
pub fn draw_spawn(key: [u32; 2], link: u32, tick: u32) -> u32 {
    philox4x32_10([link, tick, 0, STREAM_SPAWN], key)[0]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_every_random123_vector() {
        let outputs = known_answer_inputs().map(|(ctr, key)| philox4x32_10(ctr, key));
        assert_eq!(known_answers_matched(&outputs), KNOWN_ANSWERS.len());
    }

    #[test]
    fn a_wrong_output_is_not_counted() {
        let mut outputs = known_answer_inputs().map(|(ctr, key)| philox4x32_10(ctr, key));
        outputs[1][2] ^= 1;
        assert_eq!(known_answers_matched(&outputs), KNOWN_ANSWERS.len() - 1);
    }

    #[test]
    fn streams_do_not_collide() {
        let key = key_from_seed(42);
        let brake = draw_brake(key, 7, 3, 5);
        let spawn = draw_spawn(key, 7, 3);
        let route = draw_route(key, 7, 3, 5)[0];
        assert_ne!(brake, spawn);
        assert_ne!(brake, route);
    }
}
