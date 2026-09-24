//! Work-stealing backend: each phase is a rayon parallel iterator over blocks of nodes, in a pool of its own.

use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::time::Instant;

use ::rayon::prelude::*;
use ::rayon::{ThreadPool, ThreadPoolBuilder};

use super::{Engine, PhaseStats, TickStats, block_count, ns_since, run_nodes};
use crate::city::{City, NodeStep, step_a, step_b};

/// Owner value of a block that has not run yet.
const NO_WORKER: u32 = u32::MAX;

/// An atomic counter aligned to 128 bytes: one cache line, or adjacent-line prefetch pair, per counter.
#[repr(align(128))]
#[derive(Default)]
struct Padded(AtomicU64);

/// Rayon backend over a pool of `threads` workers, splitting each phase into blocks of `block` nodes.
pub struct Rayon {
    pool: ThreadPool,
    threads: usize,
    block: usize,
    busy: Vec<Padded>,
    migrated: Vec<Padded>,
    owner: [Vec<AtomicU32>; 2],
}

impl Rayon {
    /// A backend with its own pool of `threads` workers.
    pub fn new(threads: usize, block: usize) -> Result<Rayon, String> {
        let pool = ThreadPoolBuilder::new()
            .num_threads(threads)
            .thread_name(|i| format!("citysim-rayon-{i}"))
            .build()
            .map_err(|e| format!("cannot start the rayon pool: {e}"))?;
        Ok(Rayon {
            pool,
            threads,
            block,
            busy: (0..threads).map(|_| Padded::default()).collect(),
            migrated: (0..threads).map(|_| Padded::default()).collect(),
            owner: [Vec::new(), Vec::new()],
        })
    }

    fn size_owners(&mut self, blocks: usize) {
        for owner in &mut self.owner {
            if owner.len() != blocks {
                *owner = (0..blocks).map(|_| AtomicU32::new(NO_WORKER)).collect();
            }
        }
    }

    /// Runs phase `phase` (0 for A, 1 for B) over every block, adding each worker's step time to `worker_busy`.
    fn phase(&self, city: &mut City, phase: usize, step: NodeStep, worker_busy: &mut [u64]) -> PhaseStats {
        let started = Instant::now();
        let (ctx, cars, meta) = city.split();
        let per_node = ctx.params.cars_per_node();
        let owner = &self.owner[phase];
        cars.par_chunks_mut(self.block * per_node).zip(meta.par_chunks_mut(self.block)).enumerate().for_each(
            |(b, (block_cars, block_meta))| {
                let block_started = Instant::now();
                run_nodes(&ctx, b * self.block, block_cars, block_meta, step);
                let worker = ::rayon::current_thread_index().unwrap_or(0);
                self.busy[worker].0.fetch_add(ns_since(block_started), Ordering::Relaxed);
                let previous = owner[b].swap(worker as u32, Ordering::Relaxed);
                if previous != NO_WORKER && previous != worker as u32 {
                    self.migrated[worker].0.fetch_add(1, Ordering::Relaxed);
                }
            },
        );
        let wall = ns_since(started);
        let busy: Vec<u64> = self.busy.iter().map(|p| p.0.swap(0, Ordering::Relaxed)).collect();
        worker_busy.iter_mut().zip(&busy).for_each(|(total, ns)| *total += ns);
        let migrated = self.migrated.iter().map(|p| p.0.swap(0, Ordering::Relaxed)).sum();
        PhaseStats::from_busy(wall, &busy, owner.len() as u64, migrated)
    }
}

impl Engine for Rayon {
    fn name(&self) -> &'static str {
        "rayon"
    }

    fn threads(&self) -> usize {
        self.threads
    }

    fn run_ticks(&mut self, city: &mut City, ticks: u32) -> Result<TickStats, String> {
        self.size_owners(block_count(city.params.nodes(), self.block));
        let this = &*self;
        Ok(this.pool.install(|| {
            let mut stats = TickStats::default();
            let mut worker_busy = vec![0; this.threads];
            for _ in 0..ticks {
                let a = this.phase(city, 0, step_a, &mut worker_busy);
                let b = this.phase(city, 1, step_b, &mut worker_busy);
                city.tick += 1;
                stats.add(TickStats::from_phases(a, b));
            }
            TickStats::from_call(stats, &worker_busy)
        }))
    }
}
