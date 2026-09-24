//! City state and the per-intersection work of one tick.
//!
//! Every intersection owns its four approach links, its signal and its
//! counters. A tick is two phases with a barrier after each:
//!
//! * Phase A, `step_a`: move the cars on the node's own approaches, decide who
//!   crosses, remove the crossing cars and publish them in the node's outbox.
//! * Phase B, `step_b`: pull the car, if any, that each upstream neighbour
//!   published for this node, spawn new cars, publish the free entry cells of
//!   each approach, and set the signal for the next tick.
//!
//! A node writes only its own cars and `NodeMeta`. The only data it reads from
//! other nodes lives in the `Halo`, which is written in one phase and read in
//! the other, so every backend can run the nodes of a phase in any order, on
//! any thread, and get the same bits.

use std::collections::TryReserveError;
use std::io::{self, Read, Write};
use std::sync::atomic::{AtomicU8, AtomicU64, Ordering};

use crate::car::{Car, FLAG_VALID, LEFT, PARK, RIGHT, STRAIGHT, slot};
use crate::grid::{exit_port, link_id, neighbour_node, opposite};
use crate::junction::{Want, entry_gap, pop_head, push_tail, resolve, want_of};
use crate::nasch::{HeadRequest, advance_link};
use crate::params::{Demand, Event, Params, VMAX, event_boost, spawn_table, spawn_threshold};
use crate::philox::{draw_route, draw_spawn, key_from_seed};
use crate::signal::{is_green, next_signal};

/// Per-intersection state: ring metadata of the four approaches, the signal and
/// the counters. 32 bytes with a fixed C layout, shared with the GPU.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct NodeMeta {
    pub start: [u8; 4],
    pub len: [u8; 4],
    pub phase: u8,
    pub reserved: u8,
    pub elapsed: u16,
    pub sensor: u32,
    pub spawned: u32,
    pub exited: u32,
    pub parked: u32,
    pub crossed: u32,
}

/// Data one node publishes for its neighbours. Written by the owner in one
/// phase, read by others in the next, with a barrier in between. Relaxed atomic
/// access is enough, because the barrier orders it, and on x86 it compiles to
/// plain loads and stores.
pub struct Halo {
    outbox: Vec<[AtomicU64; 2]>,
    entry_gap: Vec<AtomicU8>,
}

/// An array the city could not allocate, reported by name for the stress test.
#[derive(Debug)]
pub struct AllocFailure {
    pub array: &'static str,
    pub bytes: u64,
}

fn alloc_filled<T: Clone>(array: &'static str, count: usize, value: T) -> Result<Vec<T>, AllocFailure> {
    let fail = |_: TryReserveError| AllocFailure { array, bytes: (count * std::mem::size_of::<T>()) as u64 };
    let mut v = Vec::new();
    v.try_reserve_exact(count).map_err(fail)?;
    v.resize(count, value);
    Ok(v)
}

fn alloc_with<T>(array: &'static str, count: usize, make: impl Fn() -> T) -> Result<Vec<T>, AllocFailure> {
    let fail = |_: TryReserveError| AllocFailure { array, bytes: (count * std::mem::size_of::<T>()) as u64 };
    let mut v = Vec::new();
    v.try_reserve_exact(count).map_err(fail)?;
    v.extend((0..count).map(|_| make()));
    Ok(v)
}

impl Halo {
    /// An empty halo: no cars in any outbox and every approach free.
    pub fn new(nodes: usize, link_cells: u32) -> Result<Halo, AllocFailure> {
        Ok(Halo {
            outbox: alloc_with("halo outbox", nodes * 4, || [AtomicU64::new(0), AtomicU64::new(0)])?,
            entry_gap: alloc_with("halo entry gaps", nodes * 4, || AtomicU8::new(link_cells as u8))?,
        })
    }

    /// The car node `node` handed out through `port` this tick, if any.
    pub fn outbox(&self, node: usize, port: usize) -> Option<Car> {
        let cell = &self.outbox[4 * node + port];
        let car = Car::unpack([cell[0].load(Ordering::Relaxed), cell[1].load(Ordering::Relaxed)]);
        (car.flags & FLAG_VALID != 0).then_some(car)
    }

    /// Writes all four outbox slots of `node`. Only `node`'s owner calls this, in phase A.
    pub fn publish_outbox(&self, node: usize, cars: &[Option<Car>; 4]) {
        for (port, car) in cars.iter().enumerate() {
            let words = car.map_or([0, 0], |c| Car { flags: FLAG_VALID, ..c }.pack());
            let cell = &self.outbox[4 * node + port];
            cell[0].store(words[0], Ordering::Relaxed);
            cell[1].store(words[1], Ordering::Relaxed);
        }
    }

    /// Free cells at the start of `link`, as published in the last phase B.
    pub fn entry_gap(&self, link: usize) -> u32 {
        u32::from(self.entry_gap[link].load(Ordering::Relaxed))
    }

    /// Writes the entry gaps of `node`'s four approaches. Only `node`'s owner calls this, in phase B.
    pub fn publish_entry_gaps(&self, node: usize, gaps: [u8; 4]) {
        for (d, gap) in gaps.into_iter().enumerate() {
            self.entry_gap[link_id(node, d)].store(gap, Ordering::Relaxed);
        }
    }
}

/// The signature of `step_a` and `step_b`, the two phases every backend schedules.
pub type NodeStep = fn(&StepCtx, usize, &mut [Car], &mut NodeMeta);

/// Read-only inputs to one tick, shared by every node.
#[derive(Clone, Copy)]
pub struct StepCtx<'a> {
    pub params: &'a Params,
    pub key: [u32; 2],
    pub tick: u32,
    pub spawn_base: &'a [u32],
    pub event: Option<&'a Event>,
    pub halo: &'a Halo,
}

/// The whole city.
pub struct City {
    pub params: Params,
    pub demand: Demand,
    pub spawn_base: Vec<u32>,
    pub cars: Vec<Car>,
    pub meta: Vec<NodeMeta>,
    pub halo: Halo,
    pub tick: u32,
}

/// Bytes the city needs per intersection, for sizing workloads: cars, metadata, halo and spawn table.
pub fn bytes_per_node(link_cells: u32) -> u64 {
    let cars = 4 * u64::from(link_cells) * std::mem::size_of::<Car>() as u64;
    let meta = std::mem::size_of::<NodeMeta>() as u64;
    let halo = 4 * (16 + 1);
    let spawn = 4 * 4;
    cars + meta + halo + spawn
}

impl City {
    /// An empty city at `params.start_tick`. Every large array is reserved with
    /// `try_reserve_exact`, so a city too big for memory fails here with the
    /// name and size of the array instead of being killed later.
    pub fn new(params: Params, demand: Demand) -> Result<City, AllocFailure> {
        let nodes = params.nodes();
        let cars = alloc_filled("car slots", nodes * params.cars_per_node(), Car::default())?;
        let meta = alloc_with("node metadata", nodes, NodeMeta::default)?;
        let halo = Halo::new(nodes, params.link_cells)?;
        let spawn_base = spawn_table(&params, &demand);
        let mut city = City { tick: params.start_tick, params, demand, spawn_base, cars, meta, halo };
        city.reset_signals();
        Ok(city)
    }

    fn reset_signals(&mut self) {
        let cols = self.params.cols as usize;
        for (n, meta) in self.meta.iter_mut().enumerate() {
            meta.phase = ((n / cols + n % cols) % 2) as u8;
        }
    }

    /// The read-only context of the tick about to run, together with the car and
    /// metadata arrays the nodes write. The borrows are of different fields, so the
    /// compiler knows the step functions cannot write anything the context reads.
    pub fn split(&mut self) -> (StepCtx<'_>, &mut [Car], &mut [NodeMeta]) {
        let ctx = StepCtx {
            params: &self.params,
            key: key_from_seed(self.params.seed),
            tick: self.tick,
            spawn_base: &self.spawn_base,
            event: self.demand.event.as_ref(),
            halo: &self.halo,
        };
        (ctx, &mut self.cars, &mut self.meta)
    }

    /// The car slots of `node`, its four approaches in order.
    pub fn node_cars(&self, node: usize) -> &[Car] {
        let per_node = self.params.cars_per_node();
        &self.cars[node * per_node..(node + 1) * per_node]
    }

    /// Takes every intersection's sensor count for the window just finished and resets it.
    pub fn harvest_sensors(&mut self) -> Vec<u32> {
        self.meta.iter_mut().map(|m| std::mem::take(&mut m.sensor)).collect()
    }

    /// Publishes the entry gap of every approach from its cars, as `step_b` does.
    fn publish_all_entry_gaps(&self) {
        let cells = self.params.link_cells as usize;
        for (n, (m, cars)) in self.meta.iter().zip(self.cars.chunks(4 * cells)).enumerate() {
            let gaps = std::array::from_fn(|d| entry_gap(&cars[d * cells..(d + 1) * cells], m.start[d], m.len[d]));
            self.halo.publish_entry_gaps(n, gaps);
        }
    }

    /// Writes a snapshot: `SNAPSHOT_MAGIC`, `snapshot_fields` and the tick, then
    /// every node's metadata and every car slot, all little-endian.
    pub fn write_snapshot(&self, out: &mut impl Write) -> io::Result<()> {
        out.write_all(&SNAPSHOT_MAGIC)?;
        for (_, value) in snapshot_fields(&self.params) {
            out.write_all(&value.to_le_bytes())?;
        }
        out.write_all(&self.tick.to_le_bytes())?;
        let mut buf = Vec::with_capacity(SNAPSHOT_CHUNK * 32);
        for chunk in self.meta.chunks(SNAPSHOT_CHUNK) {
            buf.clear();
            chunk.iter().for_each(|m| buf.extend_from_slice(&meta_bytes(m)));
            out.write_all(&buf)?;
        }
        for chunk in self.cars.chunks(SNAPSHOT_CHUNK) {
            buf.clear();
            chunk.iter().flat_map(Car::pack).for_each(|word| buf.extend_from_slice(&word.to_le_bytes()));
            out.write_all(&buf)?;
        }
        Ok(())
    }

    /// A city from a snapshot written by `write_snapshot`.
    ///
    /// Every field of `snapshot_fields(&params)` must equal the snapshot's; the
    /// city starts at the snapshot's tick, which also becomes `params.start_tick`.
    pub fn read_snapshot(input: &mut impl Read, mut params: Params, demand: Demand) -> Result<City, String> {
        let bad = |e: io::Error| format!("snapshot unreadable: {e}");
        let mut magic = [0u8; 8];
        input.read_exact(&mut magic).map_err(bad)?;
        if magic != SNAPSHOT_MAGIC {
            return Err("not a citysim snapshot".into());
        }
        for (name, expected) in snapshot_fields(&params) {
            let mut word = [0u8; 8];
            input.read_exact(&mut word).map_err(bad)?;
            let stored = u64::from_le_bytes(word);
            if stored != expected {
                return Err(format!("the snapshot has {name} {stored}, the flags give {expected}"));
            }
        }
        let mut tick = [0u8; 4];
        input.read_exact(&mut tick).map_err(bad)?;
        params.start_tick = u32::from_le_bytes(tick);
        let mut city = City::new(params, demand)
            .map_err(|f| format!("cannot allocate the {} ({} bytes) for the snapshot", f.array, f.bytes))?;
        let mut buf = vec![0u8; SNAPSHOT_CHUNK * 32];
        for chunk in city.meta.chunks_mut(SNAPSHOT_CHUNK) {
            let bytes = &mut buf[..chunk.len() * 32];
            input.read_exact(bytes).map_err(bad)?;
            for (m, b) in chunk.iter_mut().zip(bytes.chunks_exact(32)) {
                *m = meta_from_bytes(b);
            }
        }
        for chunk in city.cars.chunks_mut(SNAPSHOT_CHUNK / 2) {
            let bytes = &mut buf[..chunk.len() * 16];
            input.read_exact(bytes).map_err(bad)?;
            for (car, b) in chunk.iter_mut().zip(bytes.chunks_exact(16)) {
                *car = Car::unpack([le_u64(&b[..8]), le_u64(&b[8..])]);
            }
        }
        if input.read(&mut [0u8; 1]).map_err(bad)? != 0 {
            return Err("the snapshot is longer than its city".into());
        }
        city.publish_all_entry_gaps();
        Ok(city)
    }
}

/// First bytes of every snapshot file.
const SNAPSHOT_MAGIC: [u8; 8] = *b"CITYSIM1";

/// Nodes or cars converted per buffered write or read of a snapshot.
const SNAPSHOT_CHUNK: usize = 1 << 16;

/// The parameters a snapshot records, by name, in file order.
fn snapshot_fields(p: &Params) -> [(&'static str, u64); 11] {
    [
        ("rows", u64::from(p.rows)),
        ("cols", u64::from(p.cols)),
        ("link cells", u64::from(p.link_cells)),
        ("seed", p.seed),
        ("brake threshold", u64::from(p.brake)),
        ("straight threshold", u64::from(p.turn_straight)),
        ("straight-or-left threshold", u64::from(p.turn_straight_left)),
        ("trip-end threshold", u64::from(p.trip_end)),
        ("min green", u64::from(p.min_green)),
        ("max green", u64::from(p.max_green)),
        ("detector", u64::from(p.detector)),
    ]
}

fn le_u64(bytes: &[u8]) -> u64 {
    u64::from_le_bytes(bytes.try_into().expect("8 bytes"))
}

fn le_u32(bytes: &[u8]) -> u32 {
    u32::from_le_bytes(bytes.try_into().expect("4 bytes"))
}

/// The 32 bytes of `m` in field order, little-endian.
fn meta_bytes(m: &NodeMeta) -> [u8; 32] {
    let mut b = [0u8; 32];
    b[0..4].copy_from_slice(&m.start);
    b[4..8].copy_from_slice(&m.len);
    b[8] = m.phase;
    b[9] = m.reserved;
    b[10..12].copy_from_slice(&m.elapsed.to_le_bytes());
    for (i, v) in [m.sensor, m.spawned, m.exited, m.parked, m.crossed].into_iter().enumerate() {
        b[12 + 4 * i..16 + 4 * i].copy_from_slice(&v.to_le_bytes());
    }
    b
}

/// The inverse of `meta_bytes`.
fn meta_from_bytes(b: &[u8]) -> NodeMeta {
    let word = |i: usize| le_u32(&b[12 + 4 * i..16 + 4 * i]);
    NodeMeta {
        start: b[0..4].try_into().expect("4 bytes"),
        len: b[4..8].try_into().expect("4 bytes"),
        phase: b[8],
        reserved: b[9],
        elapsed: u16::from_le_bytes([b[10], b[11]]),
        sensor: word(0),
        spawned: word(1),
        exited: word(2),
        parked: word(3),
        crossed: word(4),
    }
}

/// Free entry cells behind port `port` of `node`: the target link's published
/// gap, or `VMAX` when the port leads out of the city.
fn port_room(ctx: &StepCtx, node: usize, port: usize) -> u32 {
    match neighbour_node(ctx.params.rows, ctx.params.cols, node, port) {
        None => VMAX,
        Some(next) => ctx.halo.entry_gap(link_id(next, opposite(port))),
    }
}

/// Cells the head car of approach `d` may use past its stop line.
fn head_extra(ctx: &StepCtx, node: usize, d: usize, head: &Car, green: bool) -> u32 {
    if head.turn == PARK {
        return VMAX;
    }
    if !green {
        return 0;
    }
    port_room(ctx, node, want_of(d, head).port)
}

/// Turn of the head car of approach `d` for this tick.
///
/// A car stopped at the stop line of a green approach whose target link has no
/// free entry cell takes the first of straight, left and right whose target has
/// one, and keeps its turn when none has. Reads only the halo entry gaps.
fn divert(ctx: &StepCtx, node: usize, d: usize, head: &Car, green: bool) -> u8 {
    let at_line = u32::from(head.pos) + 1 == ctx.params.link_cells;
    if !green || head.turn == PARK || head.vel != 0 || !at_line {
        return head.turn;
    }
    if port_room(ctx, node, want_of(d, head).port) > 0 {
        return head.turn;
    }
    [STRAIGHT, LEFT, RIGHT]
        .into_iter()
        .find(|&turn| turn != head.turn && port_room(ctx, node, exit_port(d, turn)) > 0)
        .unwrap_or(head.turn)
}

/// Phase A for one intersection: move, resolve, remove crossing cars, publish the outbox.
/// `cars` is this node's `4 * link_cells` car slots and `meta` its metadata.
pub fn step_a(ctx: &StepCtx, node: usize, cars: &mut [Car], meta: &mut NodeMeta) {
    let cells = ctx.params.link_cells as usize;
    let mut requests: [Option<HeadRequest>; 4] = [None; 4];
    let mut wants: [Option<Want>; 4] = [None; 4];
    for d in 0..4 {
        if meta.len[d] == 0 {
            continue;
        }
        let link = &mut cars[d * cells..(d + 1) * cells];
        let green = is_green(meta.phase, d);
        let head_slot = meta.start[d] as usize;
        link[head_slot].turn = divert(ctx, node, d, &link[head_slot], green);
        let head = link[head_slot];
        let extra = head_extra(ctx, node, d, &head, green);
        requests[d] = advance_link(link, meta.start[d], meta.len[d], extra, ctx.params.brake, ctx.key, ctx.tick);
        if requests[d].is_some() {
            wants[d] = Some(want_of(d, &head));
        }
    }
    let grants = resolve(wants);
    let mut outbox: [Option<Car>; 4] = [None; 4];
    for d in 0..4 {
        let (Some(request), Some(want)) = (requests[d], wants[d]) else { continue };
        let link = &mut cars[d * cells..(d + 1) * cells];
        if !grants[d] {
            let head = &mut link[meta.start[d] as usize];
            head.pos = (cells - 1) as u8;
            head.vel = (cells - 1) as u8 - request.from;
            continue;
        }
        let car = pop_head(link, &mut meta.start[d], &mut meta.len[d]);
        if want.turn == PARK {
            meta.parked += 1;
            continue;
        }
        meta.sensor += 1;
        meta.crossed += 1;
        match neighbour_node(ctx.params.rows, ctx.params.cols, node, want.port) {
            None => meta.exited += 1,
            Some(_) => {
                let landing = u32::from(request.from) + u32::from(request.vel) - cells as u32;
                outbox[want.port] = Some(Car { pos: landing as u8, vel: request.vel, ..car });
            }
        }
    }
    ctx.halo.publish_outbox(node, &outbox);
}

/// Turn for a car entering a link, from its route draws: trip end first, then straight, left or right.
fn route_turn(params: &Params, draws: [u32; 4]) -> u8 {
    if draws[1] < params.trip_end {
        PARK
    } else if draws[0] < params.turn_straight {
        STRAIGHT
    } else if draws[0] < params.turn_straight_left {
        LEFT
    } else {
        RIGHT
    }
}

/// Whether any car on an approach is within the detector zone before the stop line.
fn detector_occupied(link: &[Car], start: u8, len: u8, detector: u32) -> bool {
    if len == 0 {
        return false;
    }
    let from = link.len() as u32 - detector.min(link.len() as u32);
    u32::from(link[slot(start, 0, link.len())].pos) >= from
}

/// Spawn threshold of `link` at this tick, including any event near its intersection.
fn link_spawn_threshold(ctx: &StepCtx, node: usize, link: usize) -> u32 {
    let cols = ctx.params.cols as usize;
    let boost = ctx.event.map_or(0, |e| event_boost(e, (node / cols) as u32, (node % cols) as u32, ctx.tick));
    spawn_threshold(ctx.spawn_base[link], ctx.tick, boost)
}

/// Phase B for one intersection: absorb incoming cars or spawn, publish entry gaps, set the next signal.
pub fn step_b(ctx: &StepCtx, node: usize, cars: &mut [Car], meta: &mut NodeMeta) {
    let params = ctx.params;
    let cells = params.link_cells as usize;
    let mut gaps = [0u8; 4];
    for d in 0..4 {
        let link = &mut cars[d * cells..(d + 1) * cells];
        let upstream = neighbour_node(params.rows, params.cols, node, d);
        let incoming = upstream.and_then(|u| ctx.halo.outbox(u, opposite(d)));
        let id = link_id(node, d) as u32;
        if let Some(car) = incoming {
            let hop = car.hop + 1;
            let turn = route_turn(params, draw_route(ctx.key, car.spawn_link(), car.spawn_tick(), hop));
            push_tail(link, meta.start[d], &mut meta.len[d], Car { hop, turn, flags: 0, ..car });
        } else if entry_gap(link, meta.start[d], meta.len[d]) > 0
            && draw_spawn(ctx.key, id, ctx.tick) < link_spawn_threshold(ctx, node, id as usize)
        {
            let turn = route_turn(params, draw_route(ctx.key, id, ctx.tick, 0));
            push_tail(link, meta.start[d], &mut meta.len[d], Car::spawned(id, ctx.tick, turn));
            meta.spawned += 1;
        }
        gaps[d] = entry_gap(link, meta.start[d], meta.len[d]);
    }
    ctx.halo.publish_entry_gaps(node, gaps);

    let occupied =
        |d: usize| detector_occupied(&cars[d * cells..(d + 1) * cells], meta.start[d], meta.len[d], params.detector);
    let green_waiting = (0..4).filter(|&d| is_green(meta.phase, d)).any(occupied);
    let red_waiting = (0..4).filter(|&d| !is_green(meta.phase, d)).any(occupied);
    let (phase, elapsed) =
        next_signal(meta.phase, meta.elapsed, green_waiting, red_waiting, params.min_green, params.max_green);
    meta.phase = phase;
    meta.elapsed = elapsed;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::{NORTH, SOUTH, WEST};
    use crate::params::test_params;

    const CENTRE: usize = 4;

    fn ctx<'a>(params: &'a Params, halo: &'a Halo) -> StepCtx<'a> {
        StepCtx { params, key: [0, 0], tick: 0, spawn_base: &[], event: None, halo }
    }

    fn fill(halo: &Halo, node: usize, approach: usize) {
        let mut gaps = [0u8; 4];
        for (d, gap) in gaps.iter_mut().enumerate() {
            *gap = halo.entry_gap(link_id(node, d)) as u8;
        }
        gaps[approach] = 0;
        halo.publish_entry_gaps(node, gaps);
    }

    fn waiting(turn: u8, vel: u8) -> Car {
        Car { pos: 31, vel, turn, ..Car::default() }
    }

    #[test]
    fn stopped_car_keeps_a_turn_with_room() {
        let params = test_params(3, 3);
        let halo = Halo::new(9, 32).unwrap();
        assert_eq!(divert(&ctx(&params, &halo), CENTRE, WEST, &waiting(STRAIGHT, 0), true), STRAIGHT);
    }

    #[test]
    fn blocked_car_takes_the_first_turn_with_room() {
        let params = test_params(3, 3);
        let halo = Halo::new(9, 32).unwrap();
        fill(&halo, 5, WEST);
        assert_eq!(divert(&ctx(&params, &halo), CENTRE, WEST, &waiting(STRAIGHT, 0), true), LEFT);
        fill(&halo, 1, SOUTH);
        assert_eq!(divert(&ctx(&params, &halo), CENTRE, WEST, &waiting(STRAIGHT, 0), true), RIGHT);
    }

    #[test]
    fn blocked_left_turn_tries_straight_first() {
        let params = test_params(3, 3);
        let halo = Halo::new(9, 32).unwrap();
        fill(&halo, 1, SOUTH);
        assert_eq!(divert(&ctx(&params, &halo), CENTRE, WEST, &waiting(LEFT, 0), true), STRAIGHT);
    }

    #[test]
    fn car_waits_when_every_street_is_full() {
        let params = test_params(3, 3);
        let halo = Halo::new(9, 32).unwrap();
        fill(&halo, 5, WEST);
        fill(&halo, 1, SOUTH);
        fill(&halo, 7, NORTH);
        assert_eq!(divert(&ctx(&params, &halo), CENTRE, WEST, &waiting(STRAIGHT, 0), true), STRAIGHT);
    }

    #[test]
    fn only_a_stopped_car_at_a_green_line_diverts() {
        let params = test_params(3, 3);
        let halo = Halo::new(9, 32).unwrap();
        fill(&halo, 5, WEST);
        let c = ctx(&params, &halo);
        assert_eq!(divert(&c, CENTRE, WEST, &waiting(STRAIGHT, 0), false), STRAIGHT);
        assert_eq!(divert(&c, CENTRE, WEST, &waiting(STRAIGHT, 2), true), STRAIGHT);
        assert_eq!(divert(&c, CENTRE, WEST, &Car { pos: 20, ..waiting(STRAIGHT, 0) }, true), STRAIGHT);
        assert_eq!(divert(&c, CENTRE, WEST, &waiting(PARK, 0), true), PARK);
    }

    #[test]
    fn a_city_exit_always_has_room() {
        let params = test_params(1, 3);
        let halo = Halo::new(3, 32).unwrap();
        fill(&halo, 2, WEST);
        assert_eq!(divert(&ctx(&params, &halo), 1, WEST, &waiting(STRAIGHT, 0), true), LEFT);
    }
}
