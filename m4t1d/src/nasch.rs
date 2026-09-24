//! The Nagel-Schreckenberg update of one single-lane link.
//!
//! The four rules of Nagel & Schreckenberg (1992) are applied to every car in
//! parallel, meaning each car's gap is measured to where the car ahead stood at
//! the start of the tick, not where it has just moved to. A red light is a
//! stopped obstacle just past the stop line, the approach TRANSIMS used, which
//! here means the head car's gap ends at the stop line.

use crate::car::{Car, slot};
use crate::params::VMAX;
use crate::philox::draw_brake;

/// The head car of a link wants to pass its stop line this tick.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HeadRequest {
    /// Its position before moving.
    pub from: u8,
    /// The speed it ended the four rules with, which carries it past the stop line.
    pub vel: u8,
}

/// Speed after the four rules for a car at speed `vel` with `gap` free cells ahead.
/// `brake_draw` is only consulted when the car is still moving after rule 2.
pub fn nasch_speed(vel: u32, gap: u32, brake: u32, brake_draw: impl FnOnce() -> u32) -> u32 {
    let accelerated = (vel + 1).min(VMAX);
    let safe = accelerated.min(gap);
    if safe > 0 && brake_draw() < brake { safe - 1 } else { safe }
}

/// Moves every car on one link by one tick.
///
/// `cars` is the link's ring storage of `link_cells` slots, holding `len` cars
/// from slot `start`, head first. `head_extra` is how many cells the head car
/// may use beyond the stop line: 0 on red, the target link's free entry cells
/// on green, and `VMAX` for a car leaving the city or parking.
///
/// Every car that stays on the link is moved in place. If the head car would
/// pass the stop line it is left untouched and returned as a request, because
/// whether it may cross is decided at the intersection.
pub fn advance_link(
    cars: &mut [Car],
    start: u8,
    len: u8,
    head_extra: u32,
    brake: u32,
    key: [u32; 2],
    tick: u32,
) -> Option<HeadRequest> {
    let capacity = cars.len();
    let stop_line = capacity as u32 - 1;
    let mut ahead_old: Option<u32> = None;
    let mut request = None;
    for i in 0..len as usize {
        let car = &mut cars[slot(start, i, capacity)];
        let x = u32::from(car.pos);
        let gap = match ahead_old {
            None => stop_line - x + head_extra,
            Some(ahead) => ahead - x - 1,
        };
        let (spawn_link, spawn_tick) = (car.spawn_link(), car.spawn_tick());
        let v = nasch_speed(u32::from(car.vel), gap, brake, || draw_brake(key, spawn_link, spawn_tick, tick));
        ahead_old = Some(x);
        if i == 0 && x + v > stop_line {
            request = Some(HeadRequest { from: x as u8, vel: v as u8 });
        } else {
            car.pos = (x + v) as u8;
            car.vel = v as u8;
        }
    }
    request
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;
    use crate::car::STRAIGHT;
    use crate::invariants::check_link;

    const KEY: [u32; 2] = [42, 0];

    /// A ring of 6 to 64 cells holding cars at distinct positions and any legal speed, head first from `start`.
    fn arb_link() -> impl Strategy<Value = (Vec<Car>, u8, u8)> {
        (6usize..=64)
            .prop_flat_map(|cells| {
                (
                    proptest::sample::subsequence((0..cells as u8).collect::<Vec<_>>(), 0..=cells),
                    proptest::collection::vec(0..=VMAX as u8, cells),
                    0..cells as u8,
                )
            })
            .prop_map(|(mut positions, speeds, start)| {
                let cells = speeds.len();
                positions.reverse();
                let mut cars = vec![Car::default(); cells];
                for (i, (&pos, &vel)) in positions.iter().zip(&speeds).enumerate() {
                    cars[slot(start, i, cells)] = Car { id: i as u64, pos, vel, ..Car::default() };
                }
                (cars, start, positions.len() as u8)
            })
    }

    fn link_with(positions: &[(u8, u8)], cells: usize) -> Vec<Car> {
        let mut cars = vec![Car::default(); cells];
        for (i, &(pos, vel)) in positions.iter().enumerate() {
            cars[i] = Car { id: i as u64, hop: 0, pos, vel, turn: STRAIGHT, flags: 0 };
        }
        cars
    }

    #[test]
    fn free_car_accelerates_to_the_red_light() {
        let mut cars = link_with(&[(0, 0)], 32);
        let mut trail = Vec::new();
        for tick in 0..10 {
            assert_eq!(advance_link(&mut cars, 0, 1, 0, 0, KEY, tick), None);
            trail.push(cars[0].pos);
        }
        assert_eq!(trail, vec![1, 3, 6, 10, 15, 20, 25, 30, 31, 31]);
        assert_eq!(cars[0].vel, 0);
    }

    #[test]
    fn car_brakes_to_the_car_ahead() {
        let mut cars = link_with(&[(10, 0), (7, 5)], 32);
        advance_link(&mut cars, 0, 2, 0, 0, KEY, 0);
        assert_eq!(cars[0].pos, 11);
        assert_eq!(cars[1].pos, 9);
        assert_eq!(cars[1].vel, 2);
    }

    #[test]
    fn follower_uses_the_leaders_old_position() {
        let mut cars = link_with(&[(10, 5), (8, 5)], 32);
        advance_link(&mut cars, 0, 2, 0, 0, KEY, 0);
        assert_eq!(cars[0].pos, 15);
        assert_eq!(cars[1].pos, 9);
    }

    #[test]
    fn certain_braking_slows_by_one() {
        let mut cars = link_with(&[(0, 3)], 32);
        advance_link(&mut cars, 0, 1, 0, u32::MAX, KEY, 0);
        assert_eq!(cars[0].vel, 3);
        assert_eq!(cars[0].pos, 3);
    }

    #[test]
    fn stopped_car_never_draws() {
        assert_eq!(nasch_speed(0, 0, u32::MAX, || panic!("drew for a car with no room")), 0);
    }

    #[test]
    fn red_light_holds_the_head_at_the_stop_line() {
        let mut cars = link_with(&[(31, 0)], 32);
        assert_eq!(advance_link(&mut cars, 0, 1, 0, 0, KEY, 0), None);
        assert_eq!(cars[0].pos, 31);
    }

    #[test]
    fn green_with_room_asks_to_cross() {
        let mut cars = link_with(&[(29, 4)], 32);
        let request = advance_link(&mut cars, 0, 1, 32, 0, KEY, 0);
        assert_eq!(request, Some(HeadRequest { from: 29, vel: 5 }));
        assert_eq!(cars[0].pos, 29);
    }

    #[test]
    fn green_with_a_full_target_holds() {
        let mut cars = link_with(&[(31, 0)], 32);
        assert_eq!(advance_link(&mut cars, 0, 1, 0, 0, KEY, 0), None);
    }

    #[test]
    fn jammed_link_stays_still() {
        let positions: Vec<(u8, u8)> = (0..8).map(|i| (31 - i, 0)).collect();
        let mut cars = link_with(&positions, 32);
        advance_link(&mut cars, 0, 8, 0, 0, KEY, 0);
        for (i, car) in cars.iter().take(8).enumerate() {
            assert_eq!(car.pos, 31 - i as u8);
            assert_eq!(car.vel, 0);
        }
    }

    #[test]
    fn ring_wraps_around_the_storage() {
        let mut cars = vec![Car::default(); 32];
        cars[31] = Car { id: 1, hop: 0, pos: 20, vel: 0, turn: STRAIGHT, flags: 0 };
        cars[0] = Car { id: 2, hop: 0, pos: 10, vel: 0, turn: STRAIGHT, flags: 0 };
        advance_link(&mut cars, 31, 2, 0, 0, KEY, 0);
        assert_eq!(cars[31].pos, 21);
        assert_eq!(cars[0].pos, 11);
    }

    proptest! {
        #[test]
        fn any_link_stays_ordered((mut cars, start, len) in arb_link(), head_extra in 0..=VMAX, brake: u32, tick: u32) {
            let before = cars.clone();
            let cells = cars.len();
            let request = advance_link(&mut cars, start, len, head_extra, brake, KEY, tick);
            prop_assert!(check_link(&cars, start, len).is_ok());
            for i in 0..len as usize {
                let (old, new) = (before[slot(start, i, cells)], cars[slot(start, i, cells)]);
                if i == 0 && request.is_some() {
                    prop_assert_eq!(old, new);
                } else {
                    prop_assert_eq!(new.pos, old.pos + new.vel);
                    prop_assert!(new.vel <= old.vel + 1);
                }
            }
        }

        #[test]
        fn a_crossing_head_lands_in_the_room_it_was_given((mut cars, start, len) in arb_link(), head_extra in 0..=VMAX, brake: u32, tick: u32) {
            let stop_line = cars.len() as u32 - 1;
            if let Some(r) = advance_link(&mut cars, start, len, head_extra, brake, KEY, tick) {
                let reach = u32::from(r.from) + u32::from(r.vel);
                prop_assert!(reach > stop_line);
                prop_assert!(reach <= stop_line + head_extra);
            }
        }
    }
}
