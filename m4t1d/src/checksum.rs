//! Checksums that prove two runs computed the same thing.
//!
//! The state checksum covers every live car and every intersection. It is a
//! wrapping sum of per-node hashes, so it comes out the same whatever order the
//! nodes are hashed in, including a GPU reduction.
//!
//! The sensor checksum is the one m2t3d and m3t3d print: FNV-1a over every
//! non-zero hourly total, walked in hour then light order, with hours counted
//! from the first hour in the data. The same citysim run fed to
//! `m3t3d/traffic_mr --verify` prints the same number.

use crate::car::{Car, slot};
use crate::city::NodeMeta;

const FNV_OFFSET: u64 = 1_469_598_103_934_665_603;
const FNV_PRIME: u64 = 1_099_511_628_211;

/// FNV-1a over the 8 little-endian bytes of `value`, the walk m2t3d's `summaryChecksum` uses.
pub fn fnv_mix(h: u64, value: u64) -> u64 {
    (0..8).fold(h, |h, b| (h ^ ((value >> (b * 8)) & 0xFF)).wrapping_mul(FNV_PRIME))
}

/// SplitMix64 finaliser, spreading each node hash before the sum.
pub fn splitmix64(x: u64) -> u64 {
    let mut z = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Hash of one intersection: its signal and counters, then each approach's live cars from head to tail.
/// Only live fields are read, so stale ring slots and padding never leak in.
pub fn node_hash(meta: &NodeMeta, cars: &[Car], cells: usize) -> u64 {
    let mut h = FNV_OFFSET;
    for value in [
        u64::from(meta.phase),
        u64::from(meta.elapsed),
        u64::from(meta.sensor),
        u64::from(meta.spawned),
        u64::from(meta.exited),
        u64::from(meta.parked),
        u64::from(meta.crossed),
    ] {
        h = fnv_mix(h, value);
    }
    for d in 0..4 {
        let link = &cars[d * cells..(d + 1) * cells];
        h = fnv_mix(h, u64::from(meta.len[d]));
        for i in 0..meta.len[d] as usize {
            let car = &link[slot(meta.start[d], i, cells)];
            h = fnv_mix(h, car.id);
            h = fnv_mix(h, u64::from(car.hop));
            h = fnv_mix(h, u64::from(car.pos) | (u64::from(car.vel) << 8) | (u64::from(car.turn) << 16));
        }
    }
    h
}

/// Order-independent checksum of the whole city state.
pub fn state_checksum(meta: &[NodeMeta], cars: &[Car], cells: usize) -> u64 {
    meta.iter()
        .zip(cars.chunks(4 * cells))
        .enumerate()
        .map(|(n, (m, c))| {
            splitmix64(node_hash(m, c, cells).wrapping_add((n as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)))
        })
        .fold(0u64, u64::wrapping_add)
}

/// Streaming version of the m2t3d/m3t3d sensor checksum. Windows arrive in
/// time order; each hour's totals are folded in once the hour is complete.
pub struct SensorChecksum {
    h: u64,
    base_hour: Option<u32>,
    hour: Option<u32>,
    totals: Vec<u64>,
}

impl SensorChecksum {
    /// A checksum over `lights` traffic lights.
    pub fn new(lights: usize) -> SensorChecksum {
        SensorChecksum { h: FNV_OFFSET, base_hour: None, hour: None, totals: vec![0; lights] }
    }

    fn flush(&mut self) {
        let (Some(base), Some(hour)) = (self.base_hour, self.hour) else { return };
        for (light, total) in self.totals.iter_mut().enumerate() {
            if *total != 0 {
                self.h = fnv_mix(self.h, u64::from(hour - base));
                self.h = fnv_mix(self.h, light as u64);
                self.h = fnv_mix(self.h, *total);
                *total = 0;
            }
        }
    }

    /// Adds the counts of the window starting at `window_start` ticks.
    pub fn add_window(&mut self, window_start: u32, counts: &[u32]) {
        let hour = window_start / crate::params::TICKS_PER_HOUR;
        if self.hour != Some(hour) {
            self.flush();
            self.hour = Some(hour);
            self.base_hour.get_or_insert(hour);
        }
        for (total, &count) in self.totals.iter_mut().zip(counts) {
            *total += u64::from(count);
        }
    }

    /// The final checksum, with the last hour folded in.
    pub fn finish(mut self) -> u64 {
        self.flush();
        self.h
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sensor_checksum_matches_m3_on_its_tiny_example() {
        let mut sum = SensorChecksum::new(5);
        let at8 = 8 * 3600;
        let at9 = 9 * 3600;
        sum.add_window(at8, &[0, 60, 15, 120, 1]);
        sum.add_window(at9, &[0, 50, 50, 0, 90]);
        assert_eq!(sum.finish(), 17_191_051_930_829_670_833);
    }

    #[test]
    fn state_checksum_ignores_order_of_summation() {
        let meta = vec![NodeMeta { spawned: 3, ..NodeMeta::default() }, NodeMeta { exited: 1, ..NodeMeta::default() }];
        let cars = vec![Car::default(); 8];
        let forward = state_checksum(&meta, &cars, 1);
        let manual = splitmix64(node_hash(&meta[1], &cars[4..8], 1).wrapping_add(0x9E37_79B9_7F4A_7C15))
            .wrapping_add(splitmix64(node_hash(&meta[0], &cars[0..4], 1)));
        assert_eq!(forward, manual);
    }
}
