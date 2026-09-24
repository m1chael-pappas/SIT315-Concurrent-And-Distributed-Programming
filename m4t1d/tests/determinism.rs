//! Every CPU backend, thread count and partition reproduces the `seq` state and sensor counts bit for bit.

use citysim::city::City;
use citysim::compare::compare;
use citysim::params::{Demand, Params, WINDOW_TICKS, threshold};
use citysim::sched::partitioned::{Cost, CutPolicy, Partitioned};
use citysim::sched::rayon::Rayon;
use citysim::sched::{Engine, Seq};
use proptest::prelude::*;

/// 07:50, so a 1,200-tick run crosses the 08:00 change of the rush-hour weight.
const START: u32 = 7 * 3600 + 50 * 60;

/// Ticks each run covers: four sensor windows.
const TICKS: u32 = 4 * WINDOW_TICKS;

fn params(rows: u32, cols: u32, seed: u64) -> Params {
    Params {
        rows,
        cols,
        link_cells: 12,
        seed,
        brake: threshold(0.25),
        turn_straight: threshold(0.7),
        turn_straight_left: threshold(0.85),
        trip_end: threshold(0.05),
        min_green: 10,
        max_green: 45,
        detector: 8,
        start_tick: START,
    }
}

/// Demand heavy enough to jam the centre of a small city.
fn demand(rows: u32, cols: u32) -> Demand {
    Demand {
        peak: threshold(0.2),
        floor: threshold(0.01),
        spread: (rows.max(cols) / 6).max(1),
        edge: threshold(0.05),
        event: None,
    }
}

fn city(rows: u32, cols: u32, seed: u64) -> City {
    City::new(params(rows, cols, seed), demand(rows, cols)).expect("a small city fits in memory")
}

/// The state checksum and sensor counts after each window of a run with `engine`.
fn trace(engine: &mut dyn Engine, mut city: City, check: bool) -> Vec<(u64, Vec<u32>)> {
    (0..TICKS / WINDOW_TICKS)
        .map(|_| {
            let window_start = city.tick;
            engine.run_window(&mut city, WINDOW_TICKS, check).expect("the run succeeds");
            let frame = engine.harvest(&mut city, window_start).expect("the harvest succeeds");
            (engine.state_checksum(&city).expect("the checksum succeeds"), frame.counts().to_vec())
        })
        .collect()
}

/// Every CPU backend configuration the tests compare with `seq`.
fn engines(threads: usize) -> Vec<Box<dyn Engine>> {
    let block = 2;
    let balanced = |cost, every| CutPolicy::Balanced { cost, every };
    vec![
        Box::new(Partitioned::new("static", threads, block, CutPolicy::Static)),
        Box::new(Partitioned::new("balanced", threads, block, balanced(Cost::Cars, 1))),
        Box::new(Partitioned::new("balanced", threads, block, balanced(Cost::Time, 1))),
        Box::new(Partitioned::new("balanced", threads, block, balanced(Cost::Time, 0))),
        Box::new(Rayon::new(threads, block).expect("the pool starts")),
    ]
}

#[test]
fn every_backend_matches_seq() {
    for (rows, cols) in [(1, 1), (3, 3), (8, 8), (17, 13)] {
        for seed in [1, 42] {
            let reference = trace(&mut Seq, city(rows, cols, seed), false);
            assert!(reference.iter().any(|(_, counts)| counts.iter().any(|&c| c > 0)), "the city has traffic");
            for threads in [1, 2, 3, 7, 16, 32] {
                for mut engine in engines(threads) {
                    let got = trace(engine.as_mut(), city(rows, cols, seed), false);
                    assert_eq!(got, reference, "{} on {threads} threads, {rows}x{cols}, seed {seed}", engine.name());
                }
            }
        }
    }
}

#[test]
fn checked_runs_match_seq() {
    let reference = trace(&mut Seq, city(8, 8, 7), true);
    for mut engine in engines(3) {
        assert_eq!(trace(engine.as_mut(), city(8, 8, 7), true), reference, "{}", engine.name());
    }
}

#[test]
fn compare_agrees_on_every_tick() {
    let (mut a, mut b) = (city(17, 13, 5), city(17, 13, 5));
    let mut rayon = Rayon::new(4, 3).expect("the pool starts");
    let outcome = compare(&mut Seq, &mut a, &mut rayon, &mut b, 600, 1).expect("both engines run");
    assert_eq!(outcome.map(|agreement| agreement.checks), Ok(600));
}

#[test]
fn a_snapshot_resumes_on_another_backend() {
    let mut original = city(9, 11, 3);
    Seq.run_ticks(&mut original, 700).expect("the run succeeds");
    let mut bytes = Vec::new();
    original.write_snapshot(&mut bytes).expect("writing to memory succeeds");
    let mut resumed = City::read_snapshot(&mut bytes.as_slice(), params(9, 11, 3), demand(9, 11)).expect("reads back");
    assert_eq!(resumed.tick, original.tick);
    Seq.run_ticks(&mut original, 500).expect("the run succeeds");
    Partitioned::new("static", 4, 2, CutPolicy::Static).run_ticks(&mut resumed, 500).expect("the run succeeds");
    assert_eq!(Seq.state_checksum(&resumed), Seq.state_checksum(&original));
}

#[test]
fn a_snapshot_rejects_other_parameters() {
    let mut bytes = Vec::new();
    city(4, 4, 1).write_snapshot(&mut bytes).expect("writing to memory succeeds");
    let error = City::read_snapshot(&mut bytes.as_slice(), params(4, 4, 2), demand(4, 4)).err();
    assert_eq!(error.as_deref(), Some("the snapshot has seed 1, the flags give 2"));
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(48))]

    #[test]
    fn any_cuts_match_seq(block in 1usize..8, raw in proptest::collection::vec(0usize..1_000, 0..9)) {
        let blocks = (17 * 13usize).div_ceil(block);
        let mut inner: Vec<usize> = raw.iter().map(|r| r % (blocks + 1)).collect();
        inner.sort_unstable();
        let cuts: Vec<usize> = std::iter::once(0).chain(inner).chain(std::iter::once(blocks)).collect();
        let threads = cuts.len() - 1;
        let mut engine = Partitioned::new("fixed", threads, block, CutPolicy::Fixed(cuts));
        let reference = trace(&mut Seq, city(17, 13, 9), false);
        prop_assert_eq!(trace(&mut engine, city(17, 13, 9), false), reference);
    }
}
