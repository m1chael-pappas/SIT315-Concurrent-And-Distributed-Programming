//! Invariants that every correct state satisfies, checked while the simulation runs.
//!
//! `check_conservation` runs after every window, and `check_state` after every
//! tick with `--check`.

use crate::car::{Car, slot};
use crate::city::{Halo, NodeMeta};
use crate::grid::link_id;
use crate::junction::entry_gap;
use crate::params::VMAX;

/// Car totals over the whole city.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Census {
    pub spawned: u64,
    pub active: u64,
    pub exited: u64,
    pub parked: u64,
    pub crossed: u64,
    pub stopped: u64,
}

/// Sums every intersection's counters, and counts the cars on the road and how many of them are stopped.
pub fn census(meta: &[NodeMeta], cars: &[Car], cells: usize) -> Census {
    meta.iter().zip(cars.chunks(4 * cells)).fold(Census::default(), |mut c, (m, node_cars)| {
        c.spawned += u64::from(m.spawned);
        c.exited += u64::from(m.exited);
        c.parked += u64::from(m.parked);
        c.crossed += u64::from(m.crossed);
        for d in 0..4 {
            let link = &node_cars[d * cells..(d + 1) * cells];
            c.active += u64::from(m.len[d]);
            c.stopped += (0..m.len[d] as usize).filter(|&i| link[slot(m.start[d], i, cells)].vel == 0).count() as u64;
        }
        c
    })
}

/// Every car ever spawned is either still on the road, has left the city, or has parked.
pub fn check_conservation(c: &Census) -> Result<(), String> {
    if c.spawned == c.active + c.exited + c.parked {
        Ok(())
    } else {
        Err(format!(
            "conservation broken: spawned {} != active {} + exited {} + parked {}",
            c.spawned, c.active, c.exited, c.parked
        ))
    }
}

/// Positions strictly decrease from head to tail (no two cars share a cell), stay
/// on the link, and speeds stay within the limit.
pub fn check_link(link: &[Car], start: u8, len: u8) -> Result<(), String> {
    let cells = link.len();
    if len as usize > cells {
        return Err(format!("{len} cars on a link of {cells} cells"));
    }
    let mut ahead: Option<u8> = None;
    for i in 0..len as usize {
        let car = &link[slot(start, i, cells)];
        if car.pos as usize >= cells {
            return Err(format!("car {:#x} at cell {} of {cells}", car.id, car.pos));
        }
        if u32::from(car.vel) > VMAX {
            return Err(format!("car {:#x} at speed {}", car.id, car.vel));
        }
        if let Some(a) = ahead
            && car.pos >= a
        {
            return Err(format!("car {:#x} at cell {} is not behind the car at {a}", car.id, car.pos));
        }
        ahead = Some(car.pos);
    }
    Ok(())
}

/// Checks every approach of every intersection, and that each published entry gap matches its link.
pub fn check_state(meta: &[NodeMeta], cars: &[Car], halo: &Halo, cells: usize) -> Result<(), String> {
    for (n, (m, node_cars)) in meta.iter().zip(cars.chunks(4 * cells)).enumerate() {
        for d in 0..4 {
            let link = &node_cars[d * cells..(d + 1) * cells];
            check_link(link, m.start[d], m.len[d]).map_err(|e| format!("node {n} approach {d}: {e}"))?;
            let published = halo.entry_gap(link_id(n, d));
            let actual = u32::from(entry_gap(link, m.start[d], m.len[d]));
            if published != actual {
                return Err(format!("node {n} approach {d}: entry gap published {published}, actual {actual}"));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn car(pos: u8, vel: u8) -> Car {
        Car { pos, vel, ..Car::default() }
    }

    #[test]
    fn ordered_link_passes() {
        let mut link = vec![Car::default(); 32];
        link[0] = car(30, 2);
        link[1] = car(20, 5);
        link[2] = car(0, 0);
        assert!(check_link(&link, 0, 3).is_ok());
    }

    #[test]
    fn car_past_the_end_fails() {
        let link = vec![car(30, 2), Car::default(), Car::default(), Car::default()];
        assert!(check_link(&link, 0, 1).is_err());
    }

    #[test]
    fn shared_cell_fails() {
        let mut link = vec![Car::default(); 32];
        link[0] = car(10, 0);
        link[1] = car(10, 0);
        assert!(check_link(&link, 0, 2).is_err());
    }

    #[test]
    fn conservation_counts_every_car() {
        let ok = Census { spawned: 10, active: 4, exited: 5, parked: 1, ..Census::default() };
        assert!(check_conservation(&ok).is_ok());
        let lost = Census { spawned: 10, active: 4, exited: 5, parked: 0, ..Census::default() };
        assert!(check_conservation(&lost).is_err());
    }
}
