//! Simulation parameters and the demand model.
//!
//! Everything the step functions read is an integer. Probabilities are stored
//! as u32 thresholds (a draw x means "yes" when x < threshold), so the CPU and
//! the GPU make every decision from the same bits.

/// Top speed in cells per tick: 5 cells of 7.5 m per second is 135 km/h, the Nagel-Schreckenberg value.
pub const VMAX: u32 = 5;

/// Ticks per sensor window: 300 one-second ticks is the 5-minute window of the M2/M3 data.
pub const WINDOW_TICKS: u32 = 300;

/// Ticks per simulated hour.
pub const TICKS_PER_HOUR: u32 = 3600;

/// Hourly demand weights in percent, the rush-hour table from m2t3d's generator (`rushWeight`).
pub const RUSH: [u32; 24] =
    [30, 22, 18, 16, 20, 35, 60, 90, 100, 85, 70, 68, 72, 70, 68, 75, 92, 100, 88, 70, 58, 48, 40, 34];

/// Converts a probability in [0, 1] to a u32 threshold. Only configuration uses floating point.
pub fn threshold(p: f64) -> u32 {
    let clamped = p.clamp(0.0, 1.0);
    if clamped >= 1.0 { u32::MAX } else { (clamped * 4_294_967_296.0).round() as u32 }
}

/// Every value a step function reads, fixed for the whole run.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Params {
    pub rows: u32,
    pub cols: u32,
    pub link_cells: u32,
    pub seed: u64,
    pub brake: u32,
    pub turn_straight: u32,
    pub turn_straight_left: u32,
    pub trip_end: u32,
    pub min_green: u32,
    pub max_green: u32,
    pub detector: u32,
    pub start_tick: u32,
}

impl Params {
    /// Number of intersections.
    pub fn nodes(&self) -> usize {
        self.rows as usize * self.cols as usize
    }

    /// Number of directed approach links, four per intersection.
    pub fn links(&self) -> usize {
        self.nodes() * 4
    }

    /// Car slots per intersection: four approaches of `link_cells` cells each.
    pub fn cars_per_node(&self) -> usize {
        4 * self.link_cells as usize
    }
}

/// Where cars come from: a downtown peak, a suburban floor, inflow at the city edge and an optional event.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Demand {
    pub peak: u32,
    pub floor: u32,
    pub spread: u32,
    pub edge: u32,
    pub event: Option<Event>,
}

/// A temporary hotspot: extra demand around one intersection for a time window.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Event {
    pub row: u32,
    pub col: u32,
    pub start_tick: u32,
    pub end_tick: u32,
    pub peak: u32,
    pub spread: u32,
}

/// Cauchy-shaped weight in [0, `scale`] of a point at squared distance `d2`
/// from a peak of squared spread `s2`: scale * s2 / (s2 + d2). Integer only, so
/// no libm function becomes part of the reproducibility contract.
fn cauchy(scale: u32, s2: u64, d2: u64) -> u64 {
    u64::from(scale) * s2 / (s2 + d2)
}

/// Squared distance in half-node units, so the centre of an even-sized grid is an integer point.
fn half_unit_dist2(r: u32, c: u32, centre_r2: i64, centre_c2: i64) -> u64 {
    let dr = 2 * i64::from(r) - centre_r2;
    let dc = 2 * i64::from(c) - centre_c2;
    (dr * dr + dc * dc) as u64
}

/// Spawn threshold of every link at 100% rush weight. Interior links spawn from
/// driveways with the downtown-peaked weight. Links that enter the city from its
/// edge add the edge inflow, since they are the only way traffic arrives from outside.
pub fn spawn_table(params: &Params, demand: &Demand) -> Vec<u32> {
    let centre_r2 = i64::from(params.rows) - 1;
    let centre_c2 = i64::from(params.cols) - 1;
    let s2 = 4 * u64::from(demand.spread.max(1)).pow(2);
    let span = demand.peak.saturating_sub(demand.floor);
    let mut table = Vec::with_capacity(params.links());
    for r in 0..params.rows {
        for c in 0..params.cols {
            let d2 = half_unit_dist2(r, c, centre_r2, centre_c2);
            let base = u64::from(demand.floor) + cauchy(span, s2, d2);
            for d in 0..4 {
                let from_edge = crate::grid::neighbour(params.rows, params.cols, r, c, d).is_none();
                let total = base + if from_edge { u64::from(demand.edge) } else { 0 };
                table.push(total.min(u64::from(u32::MAX)) as u32);
            }
        }
    }
    table
}

/// Extra spawn threshold an event adds to one intersection's links at an absolute tick.
pub fn event_boost(event: &Event, r: u32, c: u32, tick: u32) -> u32 {
    if tick < event.start_tick || tick >= event.end_tick {
        return 0;
    }
    let s2 = u64::from(event.spread.max(1)).pow(2);
    let dr = i64::from(r) - i64::from(event.row);
    let dc = i64::from(c) - i64::from(event.col);
    cauchy(event.peak, s2, (dr * dr + dc * dc) as u64) as u32
}

/// Spawn threshold for this tick: the base scaled by the hour's rush weight, plus any event.
pub fn spawn_threshold(base: u32, tick: u32, boost: u32) -> u32 {
    let hour = (tick / TICKS_PER_HOUR) % 24;
    let scaled = u64::from(base) * u64::from(RUSH[hour as usize]) / 100;
    (scaled + u64::from(boost)).min(u64::from(u32::MAX)) as u32
}

/// Parameters for unit tests: 32-cell links and no random braking.
#[cfg(test)]
pub fn test_params(rows: u32, cols: u32) -> Params {
    Params {
        rows,
        cols,
        link_cells: 32,
        seed: 1,
        brake: 0,
        turn_straight: 0,
        turn_straight_left: 0,
        trip_end: 0,
        min_green: 10,
        max_green: 45,
        detector: 8,
        start_tick: 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::{NORTH, SOUTH, link_id};

    #[test]
    fn threshold_endpoints() {
        assert_eq!(threshold(0.0), 0);
        assert_eq!(threshold(1.0), u32::MAX);
        assert_eq!(threshold(0.5), 1 << 31);
    }

    #[test]
    fn rush_scales_spawn_threshold() {
        assert_eq!(spawn_threshold(1000, 8 * TICKS_PER_HOUR, 0), 1000);
        assert_eq!(spawn_threshold(1000, 3 * TICKS_PER_HOUR, 0), 160);
        assert_eq!(spawn_threshold(1000, 3 * TICKS_PER_HOUR, 5), 165);
    }

    #[test]
    fn demand_peaks_downtown() {
        let params = test_params(9, 9);
        let demand = Demand { peak: 10_000, floor: 100, spread: 2, edge: 0, event: None };
        let table = spawn_table(&params, &demand);
        let centre = table[link_id(4 * 9 + 4, NORTH)];
        let corner = table[link_id(0, SOUTH)];
        assert!(centre > corner);
        assert_eq!(centre, 10_000);
    }

    #[test]
    fn edge_links_get_inflow() {
        let params = test_params(3, 3);
        let demand = Demand { peak: 0, floor: 0, spread: 1, edge: 777, event: None };
        let table = spawn_table(&params, &demand);
        assert_eq!(table[link_id(0, NORTH)], 777);
        assert_eq!(table[link_id(4, NORTH)], 0);
    }
}
