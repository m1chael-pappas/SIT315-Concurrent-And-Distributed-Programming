//! The car record and the ring buffer each approach link keeps its cars in.
//!
//! A car is 16 bytes with the C layout `gpu/kernels.cu` declares. A link is a
//! single lane stored as a ring of `link_cells` slots: cars enter at the tail
//! and leave from the head.

/// Turn code: continue straight through the next intersection.
pub const STRAIGHT: u8 = 0;
/// Turn code: turn left, the kerb-side turn under keep-left rules.
pub const LEFT: u8 = 1;
/// Turn code: turn right across the oncoming lane.
pub const RIGHT: u8 = 2;
/// Turn code: leave the road into a driveway at the end of the current link.
pub const PARK: u8 = 3;

/// Set on an outbox slot that holds a car.
pub const FLAG_VALID: u8 = 1;

/// One vehicle. `id` is `(spawn tick << 32) | spawn link`, unique per car.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Car {
    pub id: u64,
    pub hop: u32,
    pub pos: u8,
    pub vel: u8,
    pub turn: u8,
    pub flags: u8,
}

impl Car {
    /// A car spawned on `link` at `tick`, standing at the start of the link.
    pub fn spawned(link: u32, tick: u32, turn: u8) -> Car {
        Car { id: (u64::from(tick) << 32) | u64::from(link), hop: 0, pos: 0, vel: 0, turn, flags: 0 }
    }

    /// The link this car was spawned on.
    pub fn spawn_link(&self) -> u32 {
        self.id as u32
    }

    /// The tick this car was spawned at.
    pub fn spawn_tick(&self) -> u32 {
        (self.id >> 32) as u32
    }

    /// Two 64-bit words, the form a car takes in the shared outbox.
    pub fn pack(&self) -> [u64; 2] {
        let tail = u64::from(self.hop)
            | (u64::from(self.pos) << 32)
            | (u64::from(self.vel) << 40)
            | (u64::from(self.turn) << 48)
            | (u64::from(self.flags) << 56);
        [self.id, tail]
    }

    /// Inverse of `pack`.
    pub fn unpack(words: [u64; 2]) -> Car {
        let tail = words[1];
        Car {
            id: words[0],
            hop: tail as u32,
            pos: (tail >> 32) as u8,
            vel: (tail >> 40) as u8,
            turn: (tail >> 48) as u8,
            flags: (tail >> 56) as u8,
        }
    }
}

/// Slot index of the i-th car from the head of a ring that starts at `start`.
pub fn slot(start: u8, i: usize, capacity: usize) -> usize {
    (start as usize + i) % capacity
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn car_is_sixteen_bytes() {
        assert_eq!(std::mem::size_of::<Car>(), 16);
        assert_eq!(std::mem::align_of::<Car>(), 8);
    }

    #[test]
    fn pack_round_trips() {
        let car = Car { id: 0x1234_5678_9abc_def0, hop: 77, pos: 31, vel: 5, turn: PARK, flags: FLAG_VALID };
        assert_eq!(Car::unpack(car.pack()), car);
    }

    #[test]
    fn id_encodes_spawn_point() {
        let car = Car::spawned(4097, 21_600, LEFT);
        assert_eq!(car.spawn_link(), 4097);
        assert_eq!(car.spawn_tick(), 21_600);
    }

    #[test]
    fn ring_wraps() {
        assert_eq!(slot(30, 0, 32), 30);
        assert_eq!(slot(30, 1, 32), 31);
        assert_eq!(slot(30, 2, 32), 0);
        assert_eq!(slot(30, 5, 32), 3);
    }
}
