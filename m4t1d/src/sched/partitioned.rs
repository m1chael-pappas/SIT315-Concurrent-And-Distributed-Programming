//! Partitioned backends: one worker per contiguous range of blocks, with a spin barrier after each phase.
//!
//! Cuts are block indices: worker `k` owns blocks `cuts[k]..cuts[k + 1]`.
//! `static` uses equal block counts. `balanced` recomputes the cuts at equal
//! shares of an estimated block cost at the start of a `run_ticks` call.

use std::time::Instant;

#[cfg(loom)]
use loom::{
    hint::spin_loop,
    sync::atomic::{AtomicBool, AtomicUsize, Ordering},
    thread::yield_now,
};
#[cfg(not(loom))]
use std::{
    hint::spin_loop,
    sync::atomic::{AtomicBool, AtomicUsize, Ordering},
    thread::yield_now,
};

use super::{Engine, PhaseStats, TickStats, block_count, ns_since, run_nodes};
use crate::car::Car;
use crate::city::{City, NodeMeta, NodeStep, StepCtx, step_a, step_b};

/// Spin-loop iterations before a waiting thread starts yielding its time slice.
const SPINS_BEFORE_YIELD: u32 = 1 << 12;

/// Phase A cost of one node, in units of the phase A cost of one car.
const NODE_COST_A: u64 = 1;

/// Phase B cost of one node, in units of the phase A cost of one car.
const NODE_COST_B: u64 = 5;

/// A reusable barrier for a fixed number of threads. Waiters spin, then yield.
///
/// Every write a thread makes before `wait` happens-before every read any
/// participating thread makes after that `wait` returns.
pub struct SpinBarrier {
    parties: usize,
    arrived: AtomicUsize,
    generation: AtomicUsize,
    poisoned: AtomicBool,
}

impl SpinBarrier {
    /// A barrier that releases its waiters once `parties` threads have arrived.
    pub fn new(parties: usize) -> SpinBarrier {
        SpinBarrier {
            parties,
            arrived: AtomicUsize::new(0),
            generation: AtomicUsize::new(0),
            poisoned: AtomicBool::new(false),
        }
    }

    /// Blocks until `parties` threads have called `wait` in the current generation.
    ///
    /// # Panics
    /// When `poison` has been called.
    pub fn wait(&self) {
        let generation = self.generation.load(Ordering::Acquire);
        if self.arrived.fetch_add(1, Ordering::AcqRel) + 1 == self.parties {
            self.arrived.store(0, Ordering::Relaxed);
            self.generation.store(generation.wrapping_add(1), Ordering::Release);
            return;
        }
        let mut spins = 0;
        while self.generation.load(Ordering::Acquire) == generation {
            assert!(!self.poisoned.load(Ordering::Relaxed), "a worker thread panicked");
            if spins < SPINS_BEFORE_YIELD {
                spins += 1;
                spin_loop();
            } else {
                yield_now();
            }
        }
    }

    /// Makes every current and later `wait` panic.
    pub fn poison(&self) {
        self.poisoned.store(true, Ordering::Relaxed);
    }
}

/// Poisons a barrier when dropped during a panic.
struct PoisonOnPanic<'a>(&'a SpinBarrier);

impl Drop for PoisonOnPanic<'_> {
    fn drop(&mut self) {
        if std::thread::panicking() {
            self.0.poison();
        }
    }
}

/// Block cost estimate the balanced backend cuts by.
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum Cost {
    /// `car_costs` of the state at the cut.
    Cars,
    /// Step nanoseconds of each block in each phase over the previous `run_ticks` call; `Cars` before any call.
    Time,
}

/// How the cuts are chosen.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CutPolicy {
    /// Equal block counts.
    Static,
    /// Equal shares of `cost` from the first call, recomputed every `every` calls of `run_ticks`; never when `every` is 0.
    Balanced { cost: Cost, every: u32 },
    /// These cuts for every call.
    Fixed(Vec<usize>),
}

/// `parts + 1` cuts splitting `blocks` blocks into ranges whose sizes differ by at most one.
pub fn equal_cuts(blocks: usize, parts: usize) -> Vec<usize> {
    (0..=parts).map(|k| k * blocks / parts).collect()
}

/// `parts + 1` cuts splitting blocks of cost `cost` into ranges of near-equal total cost.
///
/// Cut `k` is the block boundary whose prefix cost is nearest to `k / parts` of
/// the total without moving back past cut `k - 1`, so each range costs at most
/// `ceil(total / parts) + max(cost)`. Returns `equal_cuts` when every cost is zero.
pub fn cost_cuts(cost: &[u64], parts: usize) -> Vec<usize> {
    let total: u128 = cost.iter().map(|&c| u128::from(c)).sum();
    if total == 0 {
        return equal_cuts(cost.len(), parts);
    }
    let mut cuts = vec![0];
    let (mut b, mut prefix) = (0, 0u128);
    for k in 1..parts {
        let target = total * k as u128 / parts as u128;
        while b < cost.len() && prefix + u128::from(cost[b]) <= target {
            prefix += u128::from(cost[b]);
            b += 1;
        }
        if b < cost.len() && prefix < target && prefix + u128::from(cost[b]) - target < target - prefix {
            prefix += u128::from(cost[b]);
            b += 1;
        }
        cuts.push(b);
    }
    cuts.push(cost.len());
    cuts
}

/// Phase A and phase B cost of each block of `block` nodes: `NODE_COST_A` per node
/// plus one per car on its approaches, and `NODE_COST_B` per node.
pub fn car_costs(meta: &[NodeMeta], block: usize) -> Vec<[u64; 2]> {
    let cars = |m: &NodeMeta| m.len.iter().map(|&l| u64::from(l)).sum::<u64>();
    meta.chunks(block)
        .map(|nodes| {
            let n = nodes.len() as u64;
            [NODE_COST_A * n + nodes.iter().map(cars).sum::<u64>(), NODE_COST_B * n]
        })
        .collect()
}

/// Prefix sums of phase `phase` of `cost`: entry `b` is the cost of blocks `0..b`.
fn prefix(cost: &[[u64; 2]], phase: usize) -> Vec<u64> {
    std::iter::once(0)
        .chain(cost.iter().scan(0, |sum, c| {
            *sum += c[phase];
            Some(*sum)
        }))
        .collect()
}

/// Busiest range's cost under `cuts`, and the sum of squared range costs, from prefix sums.
fn range_load(prefix: &[u64], cuts: &[usize]) -> (u64, u128) {
    cuts.windows(2)
        .map(|w| prefix[w[1]] - prefix[w[0]])
        .fold((0, 0), |(max, squares), c| (max.max(c), squares + u128::from(c) * u128::from(c)))
}

/// The predicted time of one tick under `cuts`: busiest phase A range plus busiest phase B
/// range, then the summed squares of both phases' range costs to break ties.
fn tick_score(prefixes: &[Vec<u64>; 2], cuts: &[usize]) -> (u64, u128) {
    let (a, b) = (range_load(&prefixes[0], cuts), range_load(&prefixes[1], cuts));
    (a.0 + b.0, a.1 + b.1)
}

/// `parts + 1` cuts over blocks with phase costs `cost` that lower `tick_score`.
///
/// Starts from `cost_cuts` of the summed phase costs, then moves one cut at a
/// time by one block while `tick_score` strictly falls. The result never scores
/// above the start.
pub fn phase_cuts(cost: &[[u64; 2]], parts: usize) -> Vec<usize> {
    let summed: Vec<u64> = cost.iter().map(|[a, b]| a + b).collect();
    let mut cuts = cost_cuts(&summed, parts);
    let prefixes = [prefix(cost, 0), prefix(cost, 1)];
    let mut best = tick_score(&prefixes, &cuts);
    let mut improved = true;
    while improved {
        improved = false;
        for k in 1..parts {
            for forward in [false, true] {
                loop {
                    let old = cuts[k];
                    let moved = if forward { old + 1 } else { old.wrapping_sub(1) };
                    if old == 0 && !forward || moved < cuts[k - 1] || moved > cuts[k + 1] {
                        break;
                    }
                    cuts[k] = moved;
                    let score = tick_score(&prefixes, &cuts);
                    if score < best {
                        best = score;
                        improved = true;
                    } else {
                        cuts[k] = old;
                        break;
                    }
                }
            }
        }
    }
    cuts
}

/// The worker that owns block `b` under `cuts`.
fn owner(cuts: &[usize], b: usize) -> usize {
    cuts.partition_point(|&c| c <= b) - 1
}

/// Blocks whose owner differs between `old` and `new`, which cover the same blocks.
pub fn moved_blocks(old: &[usize], new: &[usize]) -> u64 {
    let blocks = *new.last().unwrap_or(&0);
    (0..blocks).filter(|&b| owner(old, b) != owner(new, b)).count() as u64
}

/// Splits `items` into consecutive slices of `lengths`.
fn split_lengths<'a, T>(mut items: &'a mut [T], lengths: &[usize]) -> Vec<&'a mut [T]> {
    lengths
        .iter()
        .map(|&n| {
            let (head, tail) = std::mem::take(&mut items).split_at_mut(n);
            items = tail;
            head
        })
        .collect()
}

/// One worker's nodes: `meta` holds nodes from `first_node`, `cars` their slots,
/// `block_ns` the phase A and phase B step time of each of their blocks.
struct Share<'a> {
    first_node: usize,
    cars: &'a mut [Car],
    meta: &'a mut [NodeMeta],
    block_ns: &'a mut [[u64; 2]],
}

/// What one worker measured over a call: step time per tick and phase, and, for worker 0, phase wall times.
struct WorkerLog {
    busy: Vec<[u64; 2]>,
    wall: Vec<[u64; 2]>,
}

/// Partitioned backend: `static` or `balanced`, by its `CutPolicy`.
pub struct Partitioned {
    name: &'static str,
    threads: usize,
    block: usize,
    policy: CutPolicy,
    cuts: Vec<usize>,
    block_ns: Vec<[u64; 2]>,
    calls: u32,
}

impl Partitioned {
    /// A backend of `threads` workers over blocks of `block` nodes.
    pub fn new(name: &'static str, threads: usize, block: usize, policy: CutPolicy) -> Partitioned {
        Partitioned { name, threads, block, policy, cuts: Vec::new(), block_ns: Vec::new(), calls: 0 }
    }

    /// The cuts for the next call, or `None` to keep the current ones.
    fn next_cuts(&self, city: &City, blocks: usize) -> Option<Vec<usize>> {
        let fresh = self.cuts.len() != self.threads + 1 || self.cuts.last() != Some(&blocks);
        match &self.policy {
            CutPolicy::Fixed(cuts) => fresh.then(|| cuts.clone()),
            CutPolicy::Static => fresh.then(|| equal_cuts(blocks, self.threads)),
            CutPolicy::Balanced { cost, every } => {
                let due = fresh || (*every > 0 && self.calls % every == 0);
                due.then(|| phase_cuts(&self.block_costs(city, *cost), self.threads))
            }
        }
    }

    /// Phase A and phase B cost of each block under `cost`.
    fn block_costs(&self, city: &City, cost: Cost) -> Vec<[u64; 2]> {
        if cost == Cost::Time && self.block_ns.iter().any(|ns| ns[0] + ns[1] > 0) {
            self.block_ns.clone()
        } else {
            car_costs(&city.meta, self.block)
        }
    }

    /// Adopts `cuts`, returning the blocks that changed owner, or 0 for the first cuts.
    fn adopt(&mut self, cuts: Vec<usize>) -> u64 {
        let moved = if self.cuts.len() == cuts.len() { moved_blocks(&self.cuts, &cuts) } else { 0 };
        self.cuts = cuts;
        moved
    }
}

/// Runs phase `phase` (0 for A, 1 for B) of `step` over every block of `share`,
/// adding each block's step time to its `block_ns`; returns the total.
fn run_share(ctx: &StepCtx, share: &mut Share, block: usize, phase: usize, step: NodeStep) -> u64 {
    let per_node = ctx.params.cars_per_node();
    let blocks = share.cars.chunks_mut(block * per_node).zip(share.meta.chunks_mut(block));
    let mut total = 0;
    for (i, ((cars, meta), ns)) in blocks.zip(share.block_ns.iter_mut()).enumerate() {
        let started = Instant::now();
        run_nodes(ctx, share.first_node + i * block, cars, meta, step);
        let spent = ns_since(started);
        ns[phase] += spent;
        total += spent;
    }
    total
}

/// The loop every worker runs: `ticks` ticks of phase A, barrier, phase B, barrier.
fn work(ctx: StepCtx, mut share: Share, ticks: u32, block: usize, barrier: &SpinBarrier) -> WorkerLog {
    let _guard = PoisonOnPanic(barrier);
    let mut log = WorkerLog { busy: Vec::with_capacity(ticks as usize), wall: Vec::with_capacity(ticks as usize) };
    for t in 0..ticks {
        let ctx = StepCtx { tick: ctx.tick + t, ..ctx };
        let started = Instant::now();
        let a = run_share(&ctx, &mut share, block, 0, step_a);
        barrier.wait();
        let mid = Instant::now();
        let b = run_share(&ctx, &mut share, block, 1, step_b);
        barrier.wait();
        log.busy.push([a, b]);
        log.wall.push([(mid - started).as_nanos() as u64, ns_since(mid)]);
    }
    log
}

/// Stats of a call from every worker's log, with worker 0's phase wall times.
fn combine(logs: &[WorkerLog], blocks: u64) -> TickStats {
    let mut stats = TickStats::default();
    for t in 0..logs[0].busy.len() {
        let phase = |p: usize| {
            let busy: Vec<u64> = logs.iter().map(|log| log.busy[t][p]).collect();
            PhaseStats::from_busy(logs[0].wall[t][p], &busy, blocks, 0)
        };
        stats.add(TickStats::from_phases(phase(0), phase(1)));
    }
    let worker_busy: Vec<u64> = logs.iter().map(|log| log.busy.iter().map(|[a, b]| a + b).sum()).collect();
    TickStats::from_call(stats, &worker_busy)
}

impl Engine for Partitioned {
    fn name(&self) -> &'static str {
        self.name
    }

    fn threads(&self) -> usize {
        self.threads
    }

    fn run_ticks(&mut self, city: &mut City, ticks: u32) -> Result<TickStats, String> {
        let nodes = city.params.nodes();
        let blocks = block_count(nodes, self.block);
        let mut repartitions = 0;
        let mut moved = 0;
        if let Some(cuts) = self.next_cuts(city, blocks) {
            let first = self.cuts.is_empty();
            moved = self.adopt(cuts);
            repartitions = u32::from(!first);
        }
        self.calls += 1;
        self.block_ns = vec![[0; 2]; blocks];

        let node_at = |b: usize| (b * self.block).min(nodes);
        let node_lengths: Vec<usize> = self.cuts.windows(2).map(|w| node_at(w[1]) - node_at(w[0])).collect();
        let block_lengths: Vec<usize> = self.cuts.windows(2).map(|w| w[1] - w[0]).collect();
        let car_lengths: Vec<usize> = node_lengths.iter().map(|n| n * city.params.cars_per_node()).collect();
        let starts: Vec<usize> = self.cuts.iter().map(|&b| node_at(b)).collect();

        let block = self.block;
        let (ctx, cars, meta) = city.split();
        let shares = split_lengths(cars, &car_lengths)
            .into_iter()
            .zip(split_lengths(meta, &node_lengths))
            .zip(split_lengths(&mut self.block_ns, &block_lengths))
            .zip(starts)
            .map(|(((cars, meta), block_ns), first_node)| Share { first_node, cars, meta, block_ns });
        let barrier = SpinBarrier::new(self.threads);
        let logs: Vec<WorkerLog> = std::thread::scope(|scope| {
            let mut shares = shares;
            let own = shares.next().expect("at least one worker");
            let handles: Vec<_> =
                shares.map(|share| scope.spawn(|| work(ctx, share, ticks, block, &barrier))).collect();
            let mut logs = vec![work(ctx, own, ticks, block, &barrier)];
            logs.extend(handles.into_iter().map(|h| h.join().expect("worker thread panicked")));
            logs
        });
        city.tick += ticks;

        let mut stats = combine(&logs, blocks as u64);
        stats.migrated = 2 * moved;
        stats.repartitions = repartitions;
        Ok(stats)
    }
}

#[cfg(all(test, not(loom)))]
mod tests {
    use proptest::prelude::*;

    use super::*;
    use crate::params::{Demand, test_params, threshold};
    use crate::sched::Seq;

    /// Ticks of the two-worker test, few enough for Miri.
    const TWO_WORKER_TICKS: u32 = 12;

    #[test]
    fn equal_cuts_cover_every_block() {
        assert_eq!(equal_cuts(10, 3), vec![0, 3, 6, 10]);
        assert_eq!(equal_cuts(2, 4), vec![0, 0, 1, 1, 2]);
    }

    #[test]
    fn cost_cuts_follow_the_cost() {
        assert_eq!(cost_cuts(&[1, 1, 1, 1, 8, 8], 2), vec![0, 5, 6]);
        assert_eq!(cost_cuts(&[0, 0, 0, 0], 2), vec![0, 2, 4]);
    }

    #[test]
    fn owner_skips_empty_ranges() {
        let cuts = [0, 0, 2, 2, 4];
        assert_eq!(owner(&cuts, 0), 1);
        assert_eq!(owner(&cuts, 3), 3);
        assert_eq!(moved_blocks(&cuts, &[0, 1, 2, 3, 4]), 2);
    }

    #[test]
    fn split_lengths_partitions_in_order() {
        let mut items = [1, 2, 3, 4, 5];
        let parts = split_lengths(&mut items, &[2, 0, 3]);
        assert_eq!(parts.iter().map(|p| p.to_vec()).collect::<Vec<_>>(), vec![vec![1, 2], vec![], vec![3, 4, 5]]);
    }

    #[test]
    fn two_workers_match_seq_on_a_small_city() {
        let demand =
            Demand { peak: threshold(0.3), floor: threshold(0.05), spread: 1, edge: threshold(0.1), event: None };
        let city = || City::new(test_params(3, 3), demand.clone()).expect("a 3x3 city fits");
        let (mut reference, mut split) = (city(), city());
        Seq.run_ticks(&mut reference, TWO_WORKER_TICKS).expect("seq runs");
        let mut engine = Partitioned::new("static", 2, 2, CutPolicy::Static);
        engine.run_ticks(&mut split, TWO_WORKER_TICKS).expect("static runs");
        assert_eq!(Seq.state_checksum(&split), Seq.state_checksum(&reference));
        assert!(reference.meta.iter().any(|m| m.spawned > 0));
    }

    #[test]
    fn barrier_releases_every_generation() {
        let barrier = SpinBarrier::new(3);
        let counter = AtomicUsize::new(0);
        std::thread::scope(|s| {
            for _ in 0..3 {
                s.spawn(|| {
                    for round in 1..=50 {
                        counter.fetch_add(1, Ordering::Relaxed);
                        barrier.wait();
                        assert_eq!(counter.load(Ordering::Relaxed), 3 * round);
                        barrier.wait();
                    }
                });
            }
        });
    }

    #[test]
    fn phase_cuts_balance_each_phase() {
        let cost = [[6, 0], [0, 2], [0, 2], [0, 2]];
        assert_eq!(cost_cuts(&cost.map(|[a, b]| a + b), 2), vec![0, 1, 4]);
        assert_eq!(phase_cuts(&cost, 2), vec![0, 2, 4]);
    }

    proptest! {
        #[test]
        fn phase_cuts_never_score_worse(
            cost in proptest::collection::vec((0u64..1_000, 0u64..1_000).prop_map(|(a, b)| [a, b]), 1..120),
            parts in 1usize..24,
        ) {
            let cuts = phase_cuts(&cost, parts);
            prop_assert_eq!(cuts.len(), parts + 1);
            prop_assert_eq!(cuts[0], 0);
            prop_assert_eq!(cuts[parts], cost.len());
            prop_assert!(cuts.windows(2).all(|w| w[0] <= w[1]));
            let summed: Vec<u64> = cost.iter().map(|[a, b]| a + b).collect();
            let prefixes = [prefix(&cost, 0), prefix(&cost, 1)];
            prop_assert!(tick_score(&prefixes, &cuts) <= tick_score(&prefixes, &cost_cuts(&summed, parts)));
        }

        #[test]
        fn cost_cuts_are_ordered_and_bounded(cost in proptest::collection::vec(0u64..1_000, 1..200), parts in 1usize..40) {
            let cuts = cost_cuts(&cost, parts);
            prop_assert_eq!(cuts.len(), parts + 1);
            prop_assert_eq!(cuts[0], 0);
            prop_assert_eq!(cuts[parts], cost.len());
            prop_assert!(cuts.windows(2).all(|w| w[0] <= w[1]));
            let total: u64 = cost.iter().sum();
            let heaviest = cost.iter().copied().max().unwrap_or(0);
            for w in cuts.windows(2) {
                let part: u64 = cost[w[0]..w[1]].iter().sum();
                prop_assert!(part <= total.div_ceil(parts as u64) + heaviest);
            }
        }
    }
}
