//! Intersection logic: who crosses, and how cars join a link.
//!
//! The junction holds no cars. Each outgoing port passes at most one car per
//! tick. Head cars that want the same port rank straight, then left, then
//! right; equal turns rank by the lower approach index.

use crate::car::{Car, PARK, slot};
use crate::grid::exit_port;

/// A head car at the stop line, wanting to leave through `port`, or to park.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Want {
    pub turn: u8,
    pub port: usize,
}

/// Priority of a turn when two approaches want the same port: lower goes first.
fn rank(turn: u8) -> u8 {
    turn
}

/// Which head cars cross this tick: the best-ranked want of each port, and every parking car, which uses no port.
pub fn resolve(wants: [Option<Want>; 4]) -> [bool; 4] {
    let mut grants = [false; 4];
    let mut best: [Option<usize>; 4] = [None; 4];
    for (approach, want) in wants.iter().enumerate() {
        let Some(want) = want else { continue };
        if want.turn == PARK {
            grants[approach] = true;
            continue;
        }
        let holder = &mut best[want.port];
        let wins = match *holder {
            None => true,
            Some(other) => rank(want.turn) < rank(wants[other].map_or(u8::MAX, |w| w.turn)),
        };
        if wins {
            *holder = Some(approach);
        }
    }
    for approach in best.into_iter().flatten() {
        grants[approach] = true;
    }
    grants
}

/// The port a car on `approach` wants, from its turn code.
pub fn want_of(approach: usize, car: &Car) -> Want {
    Want { turn: car.turn, port: if car.turn == PARK { 0 } else { exit_port(approach, car.turn) } }
}

/// Removes the head car of a ring and returns it.
pub fn pop_head(cars: &mut [Car], start: &mut u8, len: &mut u8) -> Car {
    let capacity = cars.len();
    let car = cars[*start as usize];
    *start = ((*start as usize + 1) % capacity) as u8;
    *len -= 1;
    car
}

/// Appends a car behind the tail of a ring.
pub fn push_tail(cars: &mut [Car], start: u8, len: &mut u8, car: Car) {
    let capacity = cars.len();
    cars[slot(start, *len as usize, capacity)] = car;
    *len += 1;
}

/// The tail car of a non-empty ring.
pub fn tail(cars: &[Car], start: u8, len: u8) -> Car {
    cars[slot(start, len as usize - 1, cars.len())]
}

/// Free cells at the start of a link: the tail car's position, or the whole link when empty.
pub fn entry_gap(cars: &[Car], start: u8, len: u8) -> u8 {
    if len == 0 { cars.len() as u8 } else { tail(cars, start, len).pos }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;
    use crate::car::{LEFT, RIGHT, STRAIGHT};
    use crate::grid::{EAST, NORTH, SOUTH, WEST};

    fn want(approach: usize, turn: u8) -> Option<Want> {
        Some(Want { turn, port: exit_port(approach, turn) })
    }

    /// Any mix of empty approaches and head cars with any turn, parking included.
    fn arb_wants() -> impl Strategy<Value = [Option<Want>; 4]> {
        proptest::array::uniform4(proptest::option::of(0..=PARK))
            .prop_map(|turns| std::array::from_fn(|a| turns[a].map(|turn| want_of(a, &Car { turn, ..Car::default() }))))
    }

    #[test]
    fn left_beats_a_right_turn_into_the_same_port() {
        let mut wants = [None; 4];
        wants[NORTH] = want(NORTH, LEFT);
        wants[SOUTH] = want(SOUTH, RIGHT);
        assert_eq!(wants[NORTH].unwrap().port, EAST);
        assert_eq!(wants[SOUTH].unwrap().port, EAST);
        let grants = resolve(wants);
        assert!(grants[NORTH]);
        assert!(!grants[SOUTH]);
    }

    #[test]
    fn straight_beats_a_right_turn() {
        let mut wants = [None; 4];
        wants[EAST] = want(EAST, STRAIGHT);
        wants[NORTH] = want(NORTH, RIGHT);
        assert_eq!(wants[EAST].unwrap().port, WEST);
        assert_eq!(wants[NORTH].unwrap().port, WEST);
        let grants = resolve(wants);
        assert!(grants[EAST]);
        assert!(!grants[NORTH]);
    }

    #[test]
    fn equal_turns_go_to_the_lower_approach() {
        let wants =
            [Some(Want { turn: STRAIGHT, port: SOUTH }), None, None, Some(Want { turn: STRAIGHT, port: SOUTH })];
        assert_eq!(resolve(wants), [true, false, false, false]);
    }

    #[test]
    fn different_ports_all_cross() {
        let mut wants = [None; 4];
        wants[NORTH] = want(NORTH, STRAIGHT);
        wants[SOUTH] = want(SOUTH, STRAIGHT);
        assert_eq!(resolve(wants), [true, false, true, false]);
    }

    #[test]
    fn parking_needs_no_port() {
        let wants = [Some(Want { turn: PARK, port: 0 }), None, Some(Want { turn: STRAIGHT, port: 0 }), None];
        assert_eq!(resolve(wants), [true, false, true, false]);
    }

    #[test]
    fn ring_push_pop_and_gap() {
        let mut cars = vec![Car::default(); 4];
        let (mut start, mut len) = (3u8, 0u8);
        assert_eq!(entry_gap(&cars, start, len), 4);
        push_tail(&mut cars, start, &mut len, Car { pos: 2, ..Car::default() });
        push_tail(&mut cars, start, &mut len, Car { pos: 1, ..Car::default() });
        assert_eq!(entry_gap(&cars, start, len), 1);
        let head = pop_head(&mut cars, &mut start, &mut len);
        assert_eq!(head.pos, 2);
        assert_eq!((start, len), (0, 1));
        assert_eq!(tail(&cars, start, len).pos, 1);
    }

    #[test]
    fn west_approach_left_turn_goes_north() {
        assert_eq!(want(WEST, LEFT).unwrap().port, NORTH);
    }

    proptest! {
        #[test]
        fn each_port_passes_its_best_car(wants in arb_wants()) {
            let grants = resolve(wants);
            for port in 0..4 {
                let rivals: Vec<usize> =
                    (0..4).filter(|&a| wants[a].is_some_and(|w| w.turn != PARK && w.port == port)).collect();
                let best = rivals.iter().copied().min_by_key(|&a| (wants[a].map(|w| w.turn), a));
                let granted: Vec<usize> = rivals.into_iter().filter(|&a| grants[a]).collect();
                prop_assert_eq!(granted, best.into_iter().collect::<Vec<_>>());
            }
        }

        #[test]
        fn parking_is_always_granted_and_empty_approaches_never(wants in arb_wants()) {
            let grants = resolve(wants);
            for (a, want) in wants.iter().enumerate() {
                match want {
                    None => prop_assert!(!grants[a]),
                    Some(w) if w.turn == PARK => prop_assert!(grants[a]),
                    Some(_) => {}
                }
            }
        }
    }
}
