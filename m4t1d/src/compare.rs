//! Lock-step comparison of two engines advancing two copies of the same city.

use crate::car::slot;
use crate::checksum::node_hash;
use crate::city::City;
use crate::sched::Engine;

/// Both engines produced the same state checksum at every check.
#[derive(Debug, PartialEq, Eq)]
pub struct Agreement {
    pub ticks: u32,
    pub checks: u32,
    pub state: u64,
}

/// The first check where the state checksums differed, and the first node whose `node_hash` differs.
#[derive(Debug, PartialEq, Eq)]
pub struct Divergence {
    pub tick: u32,
    pub node: Option<usize>,
    pub a: u64,
    pub b: u64,
}

/// Advances `city_a` with `a` and `city_b` with `b` by `ticks` ticks, comparing
/// state checksums every `every` ticks and after the last tick.
/// Returns at the first difference, after downloading both engines' state.
pub fn compare(
    a: &mut dyn Engine,
    city_a: &mut City,
    b: &mut dyn Engine,
    city_b: &mut City,
    ticks: u32,
    every: u32,
) -> Result<Result<Agreement, Divergence>, String> {
    let (mut done, mut checks, mut state) = (0, 0, 0);
    while done < ticks {
        let step = every.max(1).min(ticks - done);
        a.run_ticks(city_a, step)?;
        b.run_ticks(city_b, step)?;
        done += step;
        checks += 1;
        let (sum_a, sum_b) = (a.state_checksum(city_a)?, b.state_checksum(city_b)?);
        if sum_a != sum_b {
            a.download(city_a)?;
            b.download(city_b)?;
            let node = first_different_node(city_a, city_b);
            return Ok(Err(Divergence { tick: city_a.tick, node, a: sum_a, b: sum_b }));
        }
        state = sum_a;
    }
    Ok(Ok(Agreement { ticks, checks, state }))
}

/// Index of the first node whose `node_hash` differs between two cities of the same size.
pub fn first_different_node(a: &City, b: &City) -> Option<usize> {
    let cells = a.params.link_cells as usize;
    (0..a.meta.len())
        .find(|&n| node_hash(&a.meta[n], a.node_cars(n), cells) != node_hash(&b.meta[n], b.node_cars(n), cells))
}

/// A node's signal, counters and live cars, head first on each approach, as text.
pub fn describe_node(city: &City, node: usize) -> String {
    let cells = city.params.link_cells as usize;
    let m = &city.meta[node];
    let mut text = format!(
        "phase {} for {} s, sensor {}, spawned {}, exited {}, parked {}, crossed {}",
        m.phase, m.elapsed, m.sensor, m.spawned, m.exited, m.parked, m.crossed
    );
    for (d, link) in city.node_cars(node).chunks(cells).enumerate() {
        let cars: Vec<String> = (0..m.len[d] as usize)
            .map(|i| &link[slot(m.start[d], i, cells)])
            .map(|c| format!("{:#x}@{}v{}t{}h{}", c.id, c.pos, c.vel, c.turn, c.hop))
            .collect();
        text += &format!("\n    approach {d}: [{}]", cars.join(" "));
    }
    text
}
