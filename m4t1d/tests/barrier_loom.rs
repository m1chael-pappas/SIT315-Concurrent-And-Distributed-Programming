//! Model checks `SpinBarrier` with loom over every interleaving loom explores.
//!
//! `RUSTFLAGS="--cfg loom" cargo test --release --test barrier_loom`

#![cfg(loom)]

use citysim::sched::partitioned::SpinBarrier;
use loom::sync::Arc;
use loom::sync::atomic::{AtomicUsize, Ordering};
use loom::thread;

/// Thread switches allowed per execution, for waiters that spin until the last thread arrives.
const MAX_BRANCHES: usize = 100_000;

/// Runs `body(i, barrier, slots)` on `parties` threads, sharing one barrier and one slot per
/// thread, in every execution with at most `preemptions` preemptions.
fn with_threads(parties: usize, preemptions: usize, body: fn(usize, &SpinBarrier, &[AtomicUsize])) {
    let mut model = loom::model::Builder::new();
    model.preemption_bound = Some(preemptions);
    model.max_branches = MAX_BRANCHES;
    model.check(move || {
        let barrier = Arc::new(SpinBarrier::new(parties));
        let slots: Arc<Vec<AtomicUsize>> = Arc::new((0..parties).map(|_| AtomicUsize::new(0)).collect());
        let handles: Vec<_> = (1..parties)
            .map(|i| {
                let (barrier, slots) = (barrier.clone(), slots.clone());
                thread::spawn(move || body(i, &barrier, &slots))
            })
            .collect();
        body(0, &barrier, &slots);
        handles.into_iter().for_each(|h| h.join().expect("no thread panics"));
    });
}

#[test]
fn relaxed_writes_before_wait_are_seen_after_it() {
    with_threads(2, 3, |i, barrier, slots| {
        slots[i].store(i + 1, Ordering::Relaxed);
        barrier.wait();
        assert_eq!(slots[1 - i].load(Ordering::Relaxed), 2 - i);
    });
}

#[test]
fn the_barrier_is_reusable() {
    with_threads(2, 3, |i, barrier, slots| {
        for round in 1..=3 {
            slots[i].store(round, Ordering::Relaxed);
            barrier.wait();
            assert_eq!(slots[1 - i].load(Ordering::Relaxed), round);
            barrier.wait();
        }
    });
}
