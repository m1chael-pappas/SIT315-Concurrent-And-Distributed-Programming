//! Backends: schedules that run `city::step_a` then `city::step_b` for every node, once per tick.
//!
//! Every phase A finishes before its phase B starts, and every phase B before
//! the next phase A. Backends differ only in which worker runs which nodes.
//! `seq` is the reference state that every other backend reproduces bit for bit.

pub mod partitioned;
pub mod rayon;

use std::time::Instant;

use crate::car::Car;
use crate::checksum::state_checksum;
use crate::city::{City, NodeMeta, NodeStep, StepCtx, step_a, step_b};
use crate::invariants::{Census, census, check_state};
use crate::sensors::{Frame, VecFrame};

/// Timing of consecutive ticks, in nanoseconds summed over the ticks.
///
/// `busy_max_ns` sums, over every phase of every tick, the step time of the
/// busiest worker; `busy_sum_ns` sums the step time of all workers.
/// `worker_max_ns` sums, over every `run_ticks` call, the largest total step
/// time of one worker in that call. `executions` counts block runs and
/// `migrated` the block runs on a different worker than that block's previous
/// run of the same phase.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TickStats {
    pub a_ns: u64,
    pub b_ns: u64,
    pub busy_max_ns: u64,
    pub busy_sum_ns: u64,
    pub worker_max_ns: u64,
    pub executions: u64,
    pub migrated: u64,
    pub repartitions: u32,
}

/// Timing of one phase of one tick: its wall time, each worker's step time, and block runs.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PhaseStats {
    pub wall_ns: u64,
    pub busy_max_ns: u64,
    pub busy_sum_ns: u64,
    pub executions: u64,
    pub migrated: u64,
}

impl PhaseStats {
    /// Stats of a phase whose workers were busy for `busy` nanoseconds each.
    pub fn from_busy(wall_ns: u64, busy: &[u64], executions: u64, migrated: u64) -> PhaseStats {
        PhaseStats {
            wall_ns,
            busy_max_ns: busy.iter().copied().max().unwrap_or(0),
            busy_sum_ns: busy.iter().sum(),
            executions,
            migrated,
        }
    }
}

impl TickStats {
    /// Stats of one tick from its phase A and phase B.
    pub fn from_phases(a: PhaseStats, b: PhaseStats) -> TickStats {
        TickStats {
            a_ns: a.wall_ns,
            b_ns: b.wall_ns,
            busy_max_ns: a.busy_max_ns + b.busy_max_ns,
            busy_sum_ns: a.busy_sum_ns + b.busy_sum_ns,
            worker_max_ns: 0,
            executions: a.executions + b.executions,
            migrated: a.migrated + b.migrated,
            repartitions: 0,
        }
    }

    /// Sums the stats of a call's ticks and sets `worker_max_ns` from each worker's total step time in the call.
    pub fn from_call(ticks: TickStats, worker_busy: &[u64]) -> TickStats {
        TickStats { worker_max_ns: worker_busy.iter().copied().max().unwrap_or(0), ..ticks }
    }

    /// Adds `other` field by field.
    pub fn add(&mut self, other: TickStats) {
        self.a_ns += other.a_ns;
        self.b_ns += other.b_ns;
        self.busy_max_ns += other.busy_max_ns;
        self.busy_sum_ns += other.busy_sum_ns;
        self.worker_max_ns += other.worker_max_ns;
        self.executions += other.executions;
        self.migrated += other.migrated;
        self.repartitions += other.repartitions;
    }

    /// Wall time of both phases.
    pub fn wall_ns(&self) -> u64 {
        self.a_ns + self.b_ns
    }

    /// Per-phase load imbalance over `threads` workers: summed busiest-worker time over summed
    /// mean worker time, per phase of every tick. 1.0 is perfect balance and the value when nothing ran.
    pub fn imbalance(&self, threads: usize) -> f64 {
        ratio(self.busy_max_ns * threads as u64, self.busy_sum_ns)
    }

    /// Per-call load imbalance over `threads` workers: summed busiest-worker total over summed
    /// mean worker total, per `run_ticks` call. 1.0 is perfect balance and the value when nothing ran.
    pub fn call_imbalance(&self, threads: usize) -> f64 {
        ratio(self.worker_max_ns * threads as u64, self.busy_sum_ns)
    }

    /// Wall time outside the busiest worker's steps: barrier waits, scheduling and timing.
    pub fn sync_ns(&self) -> u64 {
        self.wall_ns().saturating_sub(self.busy_max_ns)
    }

    /// Fraction of block runs that migrated, 0.0 when no blocks were counted.
    pub fn migration(&self) -> f64 {
        if self.executions == 0 { 0.0 } else { self.migrated as f64 / self.executions as f64 }
    }
}

/// A way of advancing a city tick by tick.
///
/// The host arrays of `city` are the state of CPU engines. A GPU engine keeps
/// the state on the device and overrides every provided method.
pub trait Engine {
    /// Short name for banners and the CSV line.
    fn name(&self) -> &'static str;

    /// Workers the engine runs on.
    fn threads(&self) -> usize;

    /// Advances `city` by `ticks` ticks.
    fn run_ticks(&mut self, city: &mut City, ticks: u32) -> Result<TickStats, String>;

    /// Copies the engine's state into the host arrays of `city`.
    fn download(&mut self, _city: &mut City) -> Result<(), String> {
        Ok(())
    }

    /// Totals over the city at its current tick.
    fn census(&mut self, city: &City) -> Result<Census, String> {
        Ok(census(&city.meta, &city.cars, city.params.link_cells as usize))
    }

    /// `checksum::state_checksum` of the city at its current tick.
    fn state_checksum(&mut self, city: &City) -> Result<u64, String> {
        Ok(state_checksum(&city.meta, &city.cars, city.params.link_cells as usize))
    }

    /// Takes every light's sensor count, resetting it, as the frame of the window that started at `window_start`.
    fn harvest(&mut self, city: &mut City, window_start: u32) -> Result<Box<dyn Frame>, String> {
        Ok(Box::new(VecFrame { window_start, counts: city.harvest_sensors() }))
    }

    /// Runs `ticks` ticks. With `check`, runs them one at a time and calls `verify` after each.
    fn run_window(&mut self, city: &mut City, ticks: u32, check: bool) -> Result<TickStats, String> {
        if !check {
            return self.run_ticks(city, ticks);
        }
        let mut stats = TickStats::default();
        for _ in 0..ticks {
            stats.add(self.run_ticks(city, 1)?);
            self.download(city)?;
            verify(city)?;
        }
        Ok(stats)
    }
}

fn ratio(numerator: u64, denominator: u64) -> f64 {
    if denominator == 0 { 1.0 } else { numerator as f64 / denominator as f64 }
}

/// Runs `step` for nodes `first..first + meta.len()`, whose car slots are `cars`.
pub fn run_nodes(ctx: &StepCtx, first: usize, cars: &mut [Car], meta: &mut [NodeMeta], step: NodeStep) {
    let per_node = ctx.params.cars_per_node();
    for (i, (node_cars, node_meta)) in cars.chunks_mut(per_node).zip(meta.iter_mut()).enumerate() {
        step(ctx, first + i, node_cars, node_meta);
    }
}

/// Blocks of `block` nodes needed to cover `nodes` nodes; the last may be short.
pub fn block_count(nodes: usize, block: usize) -> usize {
    nodes.div_ceil(block)
}

/// Nanoseconds since `start`.
pub fn ns_since(start: Instant) -> u64 {
    start.elapsed().as_nanos() as u64
}

/// Checks the invariants that hold between ticks, naming the tick on failure.
pub fn verify(city: &City) -> Result<(), String> {
    check_state(&city.meta, &city.cars, &city.halo, city.params.link_cells as usize)
        .map_err(|e| format!("tick {}: {e}", city.tick))
}

/// One thread, nodes in index order: the reference backend.
pub struct Seq;

fn seq_phase(city: &mut City, step: NodeStep) -> PhaseStats {
    let started = Instant::now();
    let (ctx, cars, meta) = city.split();
    run_nodes(&ctx, 0, cars, meta, step);
    let wall = ns_since(started);
    PhaseStats::from_busy(wall, &[wall], 0, 0)
}

impl Engine for Seq {
    fn name(&self) -> &'static str {
        "seq"
    }

    fn threads(&self) -> usize {
        1
    }

    fn run_ticks(&mut self, city: &mut City, ticks: u32) -> Result<TickStats, String> {
        let mut stats = TickStats::default();
        for _ in 0..ticks {
            let a = seq_phase(city, step_a);
            let b = seq_phase(city, step_b);
            city.tick += 1;
            stats.add(TickStats::from_phases(a, b));
        }
        Ok(TickStats::from_call(stats, &[stats.busy_sum_ns]))
    }
}
