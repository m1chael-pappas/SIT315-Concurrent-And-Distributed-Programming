//! The CUDA backend reproduces the `seq` state and sensor counts bit for bit.
//!
//! `source cuda_env.sh && cargo test --release --features cuda --test gpu`

#![cfg(feature = "cuda")]

use citysim::city::City;
use citysim::compare::compare;
use citysim::gpu::CudaEngine;
use citysim::params::{Demand, Event, Params, WINDOW_TICKS, threshold};
use citysim::sched::{Engine, Seq};
use proptest::prelude::*;

/// 07:50, so a 1,200-tick run crosses the 08:00 change of the rush-hour weight.
const START: u32 = 7 * 3600 + 50 * 60;

fn params(rows: u32, cols: u32, link_cells: u32, seed: u64) -> Params {
    Params {
        rows,
        cols,
        link_cells,
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

fn demand(rows: u32, cols: u32, event: Option<Event>) -> Demand {
    Demand {
        peak: threshold(0.2),
        floor: threshold(0.01),
        spread: (rows.max(cols) / 6).max(1),
        edge: threshold(0.05),
        event,
    }
}

fn city(rows: u32, cols: u32, seed: u64) -> City {
    City::new(params(rows, cols, 12, seed), demand(rows, cols, None)).expect("a small city fits in memory")
}

fn gpu() -> CudaEngine {
    CudaEngine::new(64).expect("the GPU opens; source cuda_env.sh first")
}

/// The state checksum and sensor counts after each of `windows` windows of a run with `engine`.
fn trace(engine: &mut dyn Engine, mut city: City, windows: u32) -> Vec<(u64, Vec<u32>)> {
    engine.prepare(&city).expect("the engine prepares");
    (0..windows)
        .map(|_| {
            let window_start = city.tick;
            engine.run_ticks(&mut city, WINDOW_TICKS).expect("the run succeeds");
            let frame = engine.harvest(&mut city, window_start).expect("the harvest succeeds");
            (engine.state_checksum(&city).expect("the checksum succeeds"), frame.counts().to_vec())
        })
        .collect()
}

#[test]
fn cuda_matches_seq_on_every_tick() {
    let (mut a, mut b) = (city(16, 16, 42), city(16, 16, 42));
    let outcome = compare(&mut Seq, &mut a, &mut gpu(), &mut b, 3_600, 1).expect("both engines run");
    assert_eq!(outcome.map(|agreement| agreement.checks), Ok(3_600));
}

#[test]
fn cuda_matches_seq_window_by_window() {
    for (rows, cols) in [(1, 1), (3, 3), (17, 13)] {
        for seed in [1, 42] {
            let reference = trace(&mut Seq, city(rows, cols, seed), 4);
            assert_eq!(trace(&mut gpu(), city(rows, cols, seed), 4), reference, "{rows}x{cols}, seed {seed}");
        }
    }
}

#[test]
fn a_city_built_on_the_device_matches_one_built_on_the_host() {
    let p = params(20, 24, 32, 7);
    let d = demand(20, 24, None);
    let reference = trace(&mut Seq, City::new(p.clone(), d.clone()).expect("fits"), 3);
    assert_eq!(trace(&mut gpu(), City::without_host_state(p, d), 3), reference);
}

#[test]
fn the_census_matches_the_host() {
    let mut on_host = city(12, 9, 3);
    let mut on_device = city(12, 9, 3);
    Seq.run_ticks(&mut on_host, 900).expect("seq runs");
    let mut engine = gpu();
    engine.run_ticks(&mut on_device, 900).expect("cuda runs");
    assert_eq!(engine.census(&on_device), Seq.census(&on_host));
}

#[test]
fn snapshots_move_between_cpu_and_gpu() {
    let mut original = city(9, 11, 3);
    Seq.run_ticks(&mut original, 700).expect("seq runs");
    let mut bytes = Vec::new();
    original.write_snapshot(&mut bytes).expect("writes");
    let mut resumed =
        City::read_snapshot(&mut bytes.as_slice(), params(9, 11, 12, 3), demand(9, 11, None)).expect("reads");
    let mut engine = gpu();
    engine.run_ticks(&mut resumed, 500).expect("cuda runs");
    Seq.run_ticks(&mut original, 500).expect("seq runs");
    engine.download(&mut resumed).expect("downloads");
    assert_eq!(Seq.state_checksum(&resumed), Seq.state_checksum(&original));

    let mut back = Vec::new();
    resumed.write_snapshot(&mut back).expect("writes");
    let mut on_cpu =
        City::read_snapshot(&mut back.as_slice(), params(9, 11, 12, 3), demand(9, 11, None)).expect("reads");
    Seq.run_ticks(&mut on_cpu, 300).expect("seq runs");
    Seq.run_ticks(&mut original, 300).expect("seq runs");
    assert_eq!(Seq.state_checksum(&on_cpu), Seq.state_checksum(&original));
}

/// Any small city with any legal parameters and demand, with or without an event.
fn arb_city() -> impl Strategy<Value = (Params, Demand)> {
    let shape = (1u32..=8, 1u32..=8, 6u32..=40, any::<u64>(), 0..=threshold(0.5), any::<u32>(), any::<u32>());
    let demand =
        (0..=threshold(0.5), 0..=threshold(0.05), 1u32..=4, 0..=threshold(0.5), any::<bool>(), 0u32..8, 0u32..8);
    (shape, demand, 0..86_400 - WINDOW_TICKS).prop_map(
        |((rows, cols, link_cells, seed, brake, a, b), (peak, floor, spread, edge, with_event, er, ec), start)| {
            let p = Params {
                brake,
                turn_straight: a.min(b),
                turn_straight_left: a.max(b),
                start_tick: start,
                ..params(rows, cols, link_cells, seed)
            };
            let event = with_event.then_some(Event {
                row: er,
                col: ec,
                start_tick: start + 60,
                end_tick: start + 200,
                peak: threshold(0.3),
                spread: 2,
            });
            (p, Demand { peak, floor, spread, edge, event })
        },
    )
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(24))]

    #[test]
    fn any_city_matches_seq((p, d) in arb_city()) {
        let reference = trace(&mut Seq, City::new(p.clone(), d.clone()).expect("fits"), 1);
        prop_assert_eq!(trace(&mut gpu(), City::new(p, d).expect("fits"), 1), reference);
    }
}
