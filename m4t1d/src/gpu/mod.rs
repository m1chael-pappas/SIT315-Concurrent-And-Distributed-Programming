//! GPU device set-up, runtime-compiled kernels and the GPU self-test.
//!
//! `kernels.cu` is compiled at start-up with `prelude()` in front of it, by
//! NVRTC, to a cubin for the architecture of device 0. This is the only module
//! allowed `unsafe`, for raw NVRTC calls and kernel launches.

use std::ffi::{CString, c_char, c_int};
use std::sync::Arc;
use std::sync::mpsc::{Receiver, Sender, channel};
use std::time::{Duration, Instant};

use cudarc::driver::sys::CUevent_flags;
use cudarc::driver::{
    CudaContext, CudaEvent, CudaFunction, CudaModule, CudaSlice, CudaStream, DeviceRepr, LaunchConfig, PinnedHostSlice,
    PushKernelArg, ValidAsZeroBits,
};
use cudarc::nvrtc::{Ptx, result, sys};

use crate::car::{Car, FLAG_VALID, LEFT, PARK, RIGHT, STRAIGHT};
use crate::checksum::{FNV_OFFSET, FNV_PRIME, GOLDEN_GAMMA, SPLITMIX_M1, SPLITMIX_M2};
use crate::city::{City, NodeMeta};
use crate::grid::{EAST, NORTH, SOUTH, WEST};
use crate::invariants::Census;
use crate::params::{Params, VMAX, event_boost, rush_weight};
use crate::philox;
use crate::sched::{Engine, PhaseStats, TickStats};
use crate::sensors::Frame;

const KERNELS: &str = include_str!("kernels.cu");

/// Launches before timing starts, so the measurement skips first-launch set-up.
const WARM_UP_LAUNCHES: u32 = 100;

/// The device and the toolchain, as the self-test reports them.
pub struct DeviceInfo {
    pub name: String,
    pub compute: (i32, i32),
    pub memory: usize,
    pub driver: (i32, i32),
    pub nvrtc: (i32, i32),
}

/// An open GPU with the kernels compiled and loaded.
pub struct Gpu {
    /// A stream from `new_stream`, so cudarc records an event after each copy and
    /// `PinnedHostSlice::as_slice` waits for it.
    stream: Arc<CudaStream>,
    module: Arc<CudaModule>,
    pub info: DeviceInfo,
    pub arch: String,
    pub compile_ms: f64,
    pub load_ms: f64,
}

fn cuda_error(e: impl std::fmt::Display) -> String {
    format!("CUDA: {e}")
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1e3
}

/// The `#define`s `kernels.cu` expects, generated from the Rust constants and struct sizes.
pub fn prelude() -> String {
    let words: [(&str, u32); 17] = [
        ("PHILOX_M0", philox::M0),
        ("PHILOX_M1", philox::M1),
        ("PHILOX_W0", philox::W0),
        ("PHILOX_W1", philox::W1),
        ("PHILOX_ROUNDS", philox::ROUNDS),
        ("STREAM_BRAKE", philox::STREAM_BRAKE),
        ("STREAM_ROUTE", philox::STREAM_ROUTE),
        ("STREAM_SPAWN", philox::STREAM_SPAWN),
        ("VMAX", VMAX),
        ("STRAIGHT", u32::from(STRAIGHT)),
        ("LEFT", u32::from(LEFT)),
        ("RIGHT", u32::from(RIGHT)),
        ("PARK", u32::from(PARK)),
        ("FLAG_VALID", u32::from(FLAG_VALID)),
        ("CAR_BYTES", std::mem::size_of::<Car>() as u32),
        ("NODE_META_BYTES", std::mem::size_of::<NodeMeta>() as u32),
        ("DEVICE_PARAMS_BYTES", std::mem::size_of::<DeviceParams>() as u32),
    ];
    let sides = [("NORTH", NORTH), ("EAST", EAST), ("SOUTH", SOUTH), ("WEST", WEST)];
    let doubles = [
        ("FNV_OFFSET", FNV_OFFSET),
        ("FNV_PRIME", FNV_PRIME),
        ("GOLDEN_GAMMA", GOLDEN_GAMMA),
        ("SPLITMIX_M1", SPLITMIX_M1),
        ("SPLITMIX_M2", SPLITMIX_M2),
    ];
    let words = words.iter().map(|(name, value)| format!("#define {name} {value:#x}u\n"));
    let sides = sides.iter().map(|(name, value)| format!("#define {name} {value:#x}u\n"));
    let doubles = doubles.iter().map(|(name, value)| format!("#define {name} {value:#x}ull\n"));
    words.chain(sides).chain(doubles).collect()
}

/// `Err` naming the fix when the CUDA driver or the NVRTC library cannot be loaded.
fn check_libraries() -> Result<(), String> {
    if !unsafe { cudarc::driver::sys::is_culib_present() } {
        return Err("libcuda.so was not found. It comes with the NVIDIA driver, in /usr/lib/wsl/lib under WSL.".into());
    }
    if !unsafe { sys::is_culib_present() } {
        return Err("libnvrtc.so was not found. Run ./setup_nvrtc.sh once, then `source cuda_env.sh`.".into());
    }
    Ok(())
}

fn driver_version() -> Result<(i32, i32), String> {
    let mut version: c_int = 0;
    unsafe { cudarc::driver::sys::cuDriverGetVersion(&mut version) }.result().map_err(cuda_error)?;
    Ok((version / 1000, (version % 1000) / 10))
}

fn nvrtc_version() -> Result<(i32, i32), String> {
    let (mut major, mut minor): (c_int, c_int) = (0, 0);
    unsafe { sys::nvrtcVersion(&mut major, &mut minor) }.result().map_err(cuda_error)?;
    Ok((major, minor))
}

/// NVRTC's compile log as text.
fn log_text(log: &[c_char]) -> String {
    let bytes: Vec<u8> = log.iter().take_while(|&&c| c != 0).map(|&c| c as u8).collect();
    String::from_utf8_lossy(&bytes).into_owned()
}

/// The cubin of a program NVRTC has created, or its compile log on failure.
fn build_cubin(prog: sys::nvrtcProgram, arch: &str) -> Result<Vec<u8>, String> {
    let options = [format!("--gpu-architecture={arch}")];
    if let Err(e) = unsafe { result::compile_program(prog, &options) } {
        let log = unsafe { result::get_program_log(prog) }.map(|l| log_text(&l)).unwrap_or_default();
        return Err(format!("NVRTC could not compile kernels.cu for {arch}: {e}\n{log}"));
    }
    let mut size = 0usize;
    unsafe { sys::nvrtcGetCUBINSize(prog, &mut size) }.result().map_err(cuda_error)?;
    let mut cubin = vec![0u8; size];
    unsafe { sys::nvrtcGetCUBIN(prog, cubin.as_mut_ptr().cast::<c_char>()) }.result().map_err(cuda_error)?;
    Ok(cubin)
}

/// Compiles `source` to a cubin for a real architecture such as `sm_89`.
fn compile_cubin(source: &str, arch: &str) -> Result<Vec<u8>, String> {
    let source = CString::new(source).map_err(|e| e.to_string())?;
    let prog = result::create_program(&source, None).map_err(cuda_error)?;
    let cubin = build_cubin(prog, arch);
    unsafe { result::destroy_program(prog) }.map_err(cuda_error)?;
    cubin
}

impl Gpu {
    /// Opens device 0, compiles the kernels for it and loads them.
    pub fn open() -> Result<Gpu, String> {
        check_libraries()?;
        let ctx = CudaContext::new(0).map_err(cuda_error)?;
        let stream = ctx.new_stream().map_err(cuda_error)?;
        let compute = ctx.compute_capability().map_err(cuda_error)?;
        let arch = format!("sm_{}{}", compute.0, compute.1);
        let started = Instant::now();
        let cubin = compile_cubin(&(prelude() + KERNELS), &arch)?;
        let compiled = Instant::now();
        let module = ctx.load_module(Ptx::from_binary(cubin)).map_err(cuda_error)?;
        let loaded = Instant::now();
        let info = DeviceInfo {
            name: ctx.name().map_err(cuda_error)?,
            compute,
            memory: ctx.total_mem().map_err(cuda_error)?,
            driver: driver_version()?,
            nvrtc: nvrtc_version()?,
        };
        Ok(Gpu { stream, module, info, arch, compile_ms: ms(compiled - started), load_ms: ms(loaded - compiled) })
    }

    fn function(&self, name: &str) -> Result<CudaFunction, String> {
        self.module.load_function(name).map_err(cuda_error)
    }

    /// Philox-4x32-10 of every counter and key pair, computed on the GPU.
    pub fn philox(&self, inputs: &[([u32; 4], [u32; 2])]) -> Result<Vec<[u32; 4]>, String> {
        let kernel = self.function("philox_batch")?;
        let ctr: Vec<u32> = inputs.iter().flat_map(|i| i.0).collect();
        let key: Vec<u32> = inputs.iter().flat_map(|i| i.1).collect();
        let ctr_dev = self.stream.clone_htod(&ctr).map_err(cuda_error)?;
        let key_dev = self.stream.clone_htod(&key).map_err(cuda_error)?;
        let mut out_dev = self.stream.alloc_zeros::<u32>(ctr.len()).map_err(cuda_error)?;
        let n = inputs.len() as u32;
        let mut launch = self.stream.launch_builder(&kernel);
        launch.arg(&ctr_dev).arg(&key_dev).arg(&mut out_dev).arg(&n);
        unsafe { launch.launch(LaunchConfig::for_num_elems(n)) }.map_err(cuda_error)?;
        let out = self.stream.clone_dtoh(&out_dev).map_err(cuda_error)?;
        Ok(out.chunks_exact(4).map(|c| [c[0], c[1], c[2], c[3]]).collect())
    }

    /// Mean time per launch of an empty kernel, over `launches` launches and the
    /// wait for the stream to drain. This is the fixed cost every kernel pays.
    pub fn launch_latency(&self, launches: u32) -> Result<Duration, String> {
        let empty = self.function("empty")?;
        let one_warp = LaunchConfig { grid_dim: (1, 1, 1), block_dim: (32, 1, 1), shared_mem_bytes: 0 };
        let launch = || unsafe { self.stream.launch_builder(&empty).launch(one_warp) }.map(drop).map_err(cuda_error);
        for _ in 0..WARM_UP_LAUNCHES {
            launch()?;
        }
        self.stream.synchronize().map_err(cuda_error)?;
        let started = Instant::now();
        for _ in 0..launches {
            launch()?;
        }
        self.stream.synchronize().map_err(cuda_error)?;
        Ok(started.elapsed() / launches)
    }
}

/// `Car` has the C layout `kernels.cu` declares; its `static_assert` checks the size.
unsafe impl DeviceRepr for Car {}
/// An all-zero `Car` is an empty slot.
unsafe impl ValidAsZeroBits for Car {}
/// `NodeMeta` has the C layout `kernels.cu` declares; its `static_assert` checks the size.
unsafe impl DeviceRepr for NodeMeta {}
/// An all-zero `NodeMeta` is a node with empty approaches and zero counters.
unsafe impl ValidAsZeroBits for NodeMeta {}

/// Model parameters as the kernels take them, by value, with the Philox key split into words.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct DeviceParams {
    rows: u32,
    cols: u32,
    link_cells: u32,
    brake: u32,
    turn_straight: u32,
    turn_straight_left: u32,
    trip_end: u32,
    min_green: u32,
    max_green: u32,
    detector: u32,
    key0: u32,
    key1: u32,
    nodes: u32,
}

/// `DeviceParams` has the C layout `kernels.cu` declares; its `static_assert` checks the size.
unsafe impl DeviceRepr for DeviceParams {}

impl DeviceParams {
    fn new(p: &Params) -> DeviceParams {
        let [key0, key1] = philox::key_from_seed(p.seed);
        DeviceParams {
            rows: p.rows,
            cols: p.cols,
            link_cells: p.link_cells,
            brake: p.brake,
            turn_straight: p.turn_straight,
            turn_straight_left: p.turn_straight_left,
            trip_end: p.trip_end,
            min_green: p.min_green,
            max_green: p.max_green,
            detector: p.detector,
            key0,
            key1,
            nodes: p.nodes() as u32,
        }
    }
}

/// Threads per block of the `census` and `state_checksum` kernels, a whole number of warps.
const REDUCE_BLOCK: u32 = 256;

/// Page-locked sensor frames that exist at once: one filling while the writer reads the other.
const FRAMES: usize = 2;

/// Device bytes per node: car slots, metadata, outbox, entry gaps, spawn bases, event boost and sensor count.
pub fn device_bytes_per_node(link_cells: u32) -> u64 {
    let car = std::mem::size_of::<Car>() as u64;
    let meta = std::mem::size_of::<NodeMeta>() as u64;
    4 * u64::from(link_cells) * car + meta + 4 * car + 4 + 4 * 4 + 4 + 4
}

/// One-dimensional launch over `n` threads in blocks of `block`.
fn launch_over(n: usize, block: u32) -> LaunchConfig {
    LaunchConfig { grid_dim: ((n as u32).div_ceil(block), 1, 1), block_dim: (block, 1, 1), shared_mem_bytes: 0 }
}

/// The city state resident on the device.
struct DeviceState {
    params: DeviceParams,
    nodes: usize,
    cars: CudaSlice<Car>,
    meta: CudaSlice<NodeMeta>,
    outbox: CudaSlice<Car>,
    gaps: CudaSlice<u8>,
    spawn_base: CudaSlice<u32>,
    boost: CudaSlice<u32>,
    counts: CudaSlice<u32>,
    totals: CudaSlice<u64>,
    event: Option<(u32, u32)>,
}

/// A window's sensor counts in page-locked memory, copied from the device on the engine's stream.
/// Dropping it returns the buffer to the engine's `FramePool`.
struct PinnedFrame {
    window_start: u32,
    buf: Option<PinnedHostSlice<u32>>,
    home: Sender<PinnedHostSlice<u32>>,
}

impl Frame for PinnedFrame {
    fn window_start(&self) -> u32 {
        self.window_start
    }

    /// Blocks until the copy has finished.
    ///
    /// # Panics
    /// When the copy failed.
    fn counts(&self) -> &[u32] {
        let buf = self.buf.as_ref().expect("a frame holds its buffer until dropped");
        buf.as_slice().expect("the sensor copy completes")
    }
}

impl Drop for PinnedFrame {
    fn drop(&mut self) {
        if let Some(buf) = self.buf.take() {
            let _ = self.home.send(buf);
        }
    }
}

/// At most `FRAMES` page-locked buffers, reused as frames are dropped.
struct FramePool {
    home: Sender<PinnedHostSlice<u32>>,
    spare: Receiver<PinnedHostSlice<u32>>,
    allocated: usize,
}

impl FramePool {
    fn new() -> FramePool {
        let (home, spare) = channel();
        FramePool { home, spare, allocated: 0 }
    }

    /// A returned buffer, a new one of `len` counts while fewer than `FRAMES` exist, or else the next one returned.
    fn take(&mut self, ctx: &Arc<CudaContext>, len: usize) -> Result<PinnedHostSlice<u32>, String> {
        if let Ok(buf) = self.spare.try_recv() {
            return Ok(buf);
        }
        if self.allocated < FRAMES {
            self.allocated += 1;
            return unsafe { ctx.alloc_pinned::<u32>(len) }.map_err(cuda_error);
        }
        self.spare.recv().map_err(|_| "the sensor frames were lost".to_string())
    }
}

/// GPU backend: the city state stays on the device between ticks. The host
/// receives census totals, state checksums and sensor frames, and the full
/// state only through `download`.
pub struct CudaEngine {
    gpu: Gpu,
    block: u32,
    state: Option<DeviceState>,
    events: Vec<CudaEvent>,
    frames: FramePool,
}

impl CudaEngine {
    /// Opens device 0 and compiles the kernels; `block` is the thread-block size of the step kernels.
    pub fn new(block: u32) -> Result<CudaEngine, String> {
        Ok(CudaEngine { gpu: Gpu::open()?, block, state: None, events: Vec::new(), frames: FramePool::new() })
    }

    fn ctx(&self) -> Arc<CudaContext> {
        self.gpu.stream.context().clone()
    }

    /// Allocates the device state of `city`, copying its host arrays when it has them
    /// and running `init_state` when it has none.
    fn upload(&self, city: &City) -> Result<DeviceState, String> {
        let p = &city.params;
        let nodes = p.nodes();
        let need = nodes as u64 * device_bytes_per_node(p.link_cells);
        let (free, total) = self.ctx().mem_get_info().map_err(cuda_error)?;
        if need > free as u64 {
            let gib = |b: u64| b as f64 / f64::from(1u32 << 30);
            return Err(format!(
                "a {}x{} city needs {:.2} GiB of GPU memory ({} B per intersection); {} has {:.2} GiB free of {:.2} GiB",
                p.rows,
                p.cols,
                gib(need),
                device_bytes_per_node(p.link_cells),
                self.gpu.info.name,
                gib(free as u64),
                gib(total as u64)
            ));
        }
        let s = &self.gpu.stream;
        let event = city.demand.event;
        let cols = p.cols as usize;
        let boost: Vec<u32> = (0..nodes)
            .map(|n| event.map_or(0, |e| event_boost(&e, (n / cols) as u32, (n % cols) as u32, e.start_tick)))
            .collect();
        let params = DeviceParams::new(p);
        let (cars, meta, gaps) = if city.has_host_state() {
            (
                s.clone_htod(&city.cars).map_err(cuda_error)?,
                s.clone_htod(&city.meta).map_err(cuda_error)?,
                s.clone_htod(&city.halo.entry_gaps()).map_err(cuda_error)?,
            )
        } else {
            let mut meta = s.alloc_zeros::<NodeMeta>(nodes).map_err(cuda_error)?;
            let mut gaps = s.alloc_zeros::<u8>(4 * nodes).map_err(cuda_error)?;
            let init = self.gpu.function("init_state")?;
            let mut launch = s.launch_builder(&init);
            launch.arg(&params).arg(&mut meta).arg(&mut gaps);
            unsafe { launch.launch(launch_over(nodes, self.block)) }.map_err(cuda_error)?;
            (s.alloc_zeros::<Car>(nodes * p.cars_per_node()).map_err(cuda_error)?, meta, gaps)
        };
        Ok(DeviceState {
            params,
            nodes,
            cars,
            meta,
            outbox: s.alloc_zeros::<Car>(4 * nodes).map_err(cuda_error)?,
            gaps,
            spawn_base: s.clone_htod(&city.spawn_base).map_err(cuda_error)?,
            boost: s.clone_htod(&boost).map_err(cuda_error)?,
            counts: s.alloc_zeros::<u32>(nodes).map_err(cuda_error)?,
            totals: s.alloc_zeros::<u64>(6).map_err(cuda_error)?,
            event: event.map(|e| (e.start_tick, e.end_tick)),
        })
    }

    /// The device state, uploaded from `city` on first use.
    fn state(&mut self, city: &City) -> Result<&mut DeviceState, String> {
        if self.state.is_none() {
            self.state = Some(self.upload(city)?);
        }
        Ok(self.state.as_mut().expect("uploaded above"))
    }

    /// Ensures at least `n` timing events exist.
    fn reserve_events(&mut self, n: usize) -> Result<(), String> {
        let ctx = self.ctx();
        while self.events.len() < n {
            self.events.push(ctx.new_event(Some(CUevent_flags::CU_EVENT_DEFAULT)).map_err(cuda_error)?);
        }
        Ok(())
    }

    /// Runs a reduction kernel over every node into the zeroed `totals` and returns them.
    fn reduce(&mut self, city: &City, kernel: &str) -> Result<Vec<u64>, String> {
        let function = self.gpu.function(kernel)?;
        let stream = self.gpu.stream.clone();
        let state = self.state(city)?;
        stream.memset_zeros(&mut state.totals).map_err(cuda_error)?;
        let mut launch = stream.launch_builder(&function);
        launch.arg(&state.params).arg(&state.cars).arg(&state.meta).arg(&mut state.totals);
        unsafe { launch.launch(launch_over(state.nodes, REDUCE_BLOCK)) }.map_err(cuda_error)?;
        stream.clone_dtoh(&state.totals).map_err(cuda_error)
    }
}

impl Engine for CudaEngine {
    fn name(&self) -> &'static str {
        "cuda"
    }

    fn threads(&self) -> usize {
        1
    }

    fn describe(&self) -> String {
        let info = &self.gpu.info;
        format!("{} ({}, {} threads per block)", info.name, self.gpu.arch, self.block)
    }

    fn compile_ms(&self) -> f64 {
        self.gpu.compile_ms
    }

    fn prepare(&mut self, city: &City) -> Result<(), String> {
        self.state(city).map(drop)
    }

    fn run_ticks(&mut self, city: &mut City, ticks: u32) -> Result<TickStats, String> {
        self.reserve_events(3 * ticks as usize)?;
        let (step_a, step_b) = (self.gpu.function("step_a")?, self.gpu.function("step_b")?);
        let stream = self.gpu.stream.clone();
        let block = self.block;
        self.state(city)?;
        let CudaEngine { state: Some(state), events, .. } = self else { unreachable!("state uploaded above") };
        let steps = launch_over(state.nodes, block);
        for t in 0..ticks {
            let tick = city.tick + t;
            let rush = rush_weight(tick);
            let event_on = u32::from(state.event.is_some_and(|(start, end)| start <= tick && tick < end));
            let marks = &events[3 * t as usize..3 * t as usize + 3];
            marks[0].record(&stream).map_err(cuda_error)?;
            let mut a = stream.launch_builder(&step_a);
            a.arg(&state.params)
                .arg(&tick)
                .arg(&mut state.cars)
                .arg(&mut state.meta)
                .arg(&mut state.outbox)
                .arg(&state.gaps);
            unsafe { a.launch(steps) }.map_err(cuda_error)?;
            marks[1].record(&stream).map_err(cuda_error)?;
            let mut b = stream.launch_builder(&step_b);
            b.arg(&state.params)
                .arg(&tick)
                .arg(&rush)
                .arg(&event_on)
                .arg(&state.spawn_base)
                .arg(&state.boost)
                .arg(&mut state.cars)
                .arg(&mut state.meta)
                .arg(&state.outbox)
                .arg(&mut state.gaps);
            unsafe { b.launch(steps) }.map_err(cuda_error)?;
            marks[2].record(&stream).map_err(cuda_error)?;
        }
        stream.synchronize().map_err(cuda_error)?;
        let (mut a_ms, mut b_ms) = (0.0f64, 0.0f64);
        for marks in events[..3 * ticks as usize].chunks_exact(3) {
            a_ms += f64::from(marks[0].elapsed_ms(&marks[1]).map_err(cuda_error)?);
            b_ms += f64::from(marks[1].elapsed_ms(&marks[2]).map_err(cuda_error)?);
        }
        city.tick += ticks;
        let (a_ns, b_ns) = ((a_ms * 1e6) as u64, (b_ms * 1e6) as u64);
        let phases = TickStats::from_phases(
            PhaseStats::from_busy(a_ns, &[a_ns], 0, 0),
            PhaseStats::from_busy(b_ns, &[b_ns], 0, 0),
        );
        Ok(TickStats::from_call(phases, &[a_ns + b_ns]))
    }

    fn download(&mut self, city: &mut City) -> Result<(), String> {
        let Some(state) = &self.state else { return Ok(()) };
        if !city.has_host_state() {
            return Err("this run keeps no host copy of the city".into());
        }
        let stream = &self.gpu.stream;
        stream.memcpy_dtoh(&state.cars, &mut city.cars).map_err(cuda_error)?;
        stream.memcpy_dtoh(&state.meta, &mut city.meta).map_err(cuda_error)?;
        city.halo.set_entry_gaps(&stream.clone_dtoh(&state.gaps).map_err(cuda_error)?);
        Ok(())
    }

    fn census(&mut self, city: &City) -> Result<Census, String> {
        let t = self.reduce(city, "census")?;
        Ok(Census { spawned: t[0], exited: t[1], parked: t[2], crossed: t[3], active: t[4], stopped: t[5] })
    }

    fn state_checksum(&mut self, city: &City) -> Result<u64, String> {
        Ok(self.reduce(city, "state_checksum")?[0])
    }

    fn harvest(&mut self, city: &mut City, window_start: u32) -> Result<Box<dyn Frame>, String> {
        let function = self.gpu.function("harvest")?;
        let (stream, ctx, block) = (self.gpu.stream.clone(), self.ctx(), self.block);
        let nodes = city.params.nodes();
        let mut buf = self.frames.take(&ctx, nodes)?;
        let home = self.frames.home.clone();
        let state = self.state(city)?;
        let mut launch = stream.launch_builder(&function);
        launch.arg(&state.params).arg(&mut state.meta).arg(&mut state.counts);
        unsafe { launch.launch(launch_over(nodes, block)) }.map_err(cuda_error)?;
        stream.memcpy_dtoh(&state.counts, &mut buf).map_err(cuda_error)?;
        Ok(Box::new(PinnedFrame { window_start, buf: Some(buf), home }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prelude_defines_every_shared_constant() {
        let prelude = prelude();
        assert!(prelude.contains("#define PHILOX_M0 0xd2511f53u\n"));
        assert!(prelude.contains("#define PHILOX_ROUNDS 0xau\n"));
        assert!(prelude.contains("#define GOLDEN_GAMMA 0x9e3779b97f4a7c15ull\n"));
        assert!(prelude.contains("#define CAR_BYTES 0x10u\n"));
        assert_eq!(prelude.lines().count(), 26);
    }
}
