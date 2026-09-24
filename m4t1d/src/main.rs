//! citysim: a deterministic parallel city traffic simulator (SIT315 M4.T1D).
//!
//! Cars move through a grid of actuated traffic lights under the
//! Nagel-Schreckenberg rules. Every backend computes bit-identical results,
//! and each light's sensor writes car counts in the m2t3d/m3t3d CSV format.
//! See README.md for the commands.

#![deny(unsafe_code)]

mod cli;

use std::fs::File;
use std::io::{BufReader, BufWriter, Write};
use std::path::Path;
use std::process::ExitCode;
use std::time::Instant;

use clap::Parser;

use citysim::city::{AllocFailure, City, bytes_per_node};
use citysim::compare::{compare, describe_node};
#[cfg(feature = "cuda")]
use citysim::gpu::{self, CudaEngine};
use citysim::invariants::check_conservation;
use citysim::params::WINDOW_TICKS;
use citysim::philox;
use citysim::sched::partitioned::{CutPolicy, Partitioned};
use citysim::sched::rayon::Rayon;
use citysim::sched::{Engine, Seq, TickStats};
use citysim::sensors::{SensorWriter, stamp};
use citysim::stats::{RunReport, count_of, thousands, window_line};

use crate::cli::{BackendKind, CityArgs, Cli, Command, CompareArgs, EngineArgs, InfoArgs, RunArgs, SpanArgs};

/// Philox counters the self-test computes on both the GPU and the CPU.
#[cfg(feature = "cuda")]
const SELFTEST_COUNTERS: u32 = 1 << 20;

/// Empty kernel launches the self-test times.
#[cfg(feature = "cuda")]
const SELFTEST_LAUNCHES: u32 = 10_000;

/// A command's failure: the message, empty when already printed, and the exit code.
type Failure = (String, u8);

fn fail(message: String) -> Failure {
    (message, 1)
}

fn gib(bytes: u64) -> f64 {
    bytes as f64 / f64::from(1u32 << 30)
}

/// Prints which array did not fit, its size, and the memory per intersection.
fn report_alloc_failure(failure: &AllocFailure, rows: u32, cols: u32, link_cells: u32) {
    let nodes = u64::from(rows) * u64::from(cols);
    let per_node = bytes_per_node(link_cells);
    eprintln!(
        "citysim: cannot allocate the {}: {} bytes ({:.2} GiB) for a {rows}x{cols} city",
        failure.array,
        failure.bytes,
        gib(failure.bytes)
    );
    eprintln!(
        "  memory per intersection: {per_node} B (4 approaches x {link_cells} cells x 16 B car slots, plus metadata and halo)"
    );
    eprintln!(
        "  whole city: about {:.2} GiB; `citysim info --city {rows}x{cols}` sizes a city before running it",
        gib(nodes * per_node)
    );
}

/// The backend `kind` configured by `args`.
fn make_engine(kind: BackendKind, args: &EngineArgs) -> Result<Box<dyn Engine>, String> {
    let threads = args.threads.unwrap_or_else(|| std::thread::available_parallelism().map_or(1, |n| n.get()));
    if threads == 0 || args.block == 0 {
        return Err("--threads and --block must be at least 1".into());
    }
    let balanced = CutPolicy::Balanced { cost: args.cost, every: args.rebalance_every };
    Ok(match kind {
        BackendKind::Seq => Box::new(Seq),
        BackendKind::Static => Box::new(Partitioned::new("static", threads, args.block, CutPolicy::Static)),
        BackendKind::Balanced => Box::new(Partitioned::new("balanced", threads, args.block, balanced)),
        BackendKind::Rayon => Box::new(Rayon::new(threads, args.block)?),
        BackendKind::Cuda => cuda_engine(args.block)?,
    })
}

#[cfg(feature = "cuda")]
fn cuda_engine(block: usize) -> Result<Box<dyn Engine>, String> {
    let block = u32::try_from(block).map_err(|_| format!("--block {block} is too large for cuda"))?;
    Ok(Box::new(CudaEngine::new(block)?))
}

#[cfg(not(feature = "cuda"))]
fn cuda_engine(_block: usize) -> Result<Box<dyn Engine>, String> {
    Err("this build has no GPU backend; rebuild with --features cuda".into())
}

/// The city at the start of `span`: read from its snapshot, or empty at its start time.
/// Without `host_state`, an empty city gets no host arrays.
fn make_city(city: &CityArgs, span: &SpanArgs, host_state: bool) -> Result<City, Failure> {
    let Some(path) = &span.load_state else {
        let params = city.params(span.start);
        if !host_state {
            return Ok(City::without_host_state(params, city.demand()));
        }
        return City::new(params.clone(), city.demand()).map_err(|f| {
            report_alloc_failure(&f, params.rows, params.cols, params.link_cells);
            (String::new(), 2)
        });
    };
    let file = File::open(path).map_err(|e| fail(format!("cannot open {}: {e}", path.display())))?;
    City::read_snapshot(&mut BufReader::with_capacity(8 << 20, file), city.params(0), city.demand())
        .map_err(|e| fail(format!("{}: {e}", path.display())))
}

fn save_snapshot(city: &City, path: &Path) -> Result<(), Failure> {
    let write = || -> std::io::Result<()> {
        let mut out = BufWriter::with_capacity(8 << 20, File::create(path)?);
        city.write_snapshot(&mut out)?;
        out.flush()
    };
    write().map_err(|e| fail(format!("cannot write the snapshot {}: {e}", path.display())))
}

fn banner(city: &City, engine: &dyn Engine, duration: u32) {
    let p = &city.params;
    eprintln!(
        "[citysim] {}x{} city, {} intersections, backend {} on {}, {} to {}",
        p.rows,
        p.cols,
        thousands(p.nodes() as u64),
        engine.name(),
        engine.describe(),
        stamp(city.tick),
        &stamp(city.tick + duration)[11..]
    );
}

fn run(args: RunArgs) -> Result<(), Failure> {
    let mut engine = make_engine(args.backend, &args.engine).map_err(fail)?;
    let host_state = args.backend != BackendKind::Cuda || args.check || args.save_state.is_some();
    let mut city = make_city(&args.city, &args.span, host_state)?;
    banner(&city, engine.as_ref(), args.span.duration);
    engine.prepare(&city).map_err(|e| (e, 2))?;

    let mut writer = SensorWriter::start(args.sensors.clone(), city.params.nodes(), 2);
    let mut log = args
        .window_log
        .as_ref()
        .map(|p| File::create(p).map(BufWriter::new))
        .transpose()
        .map_err(|e| fail(format!("cannot create the window log: {e}")))?;

    let threads = engine.threads();
    let mut totals = TickStats::default();
    let mut sim_ms = 0.0;
    let mut last = engine.census(&city).map_err(fail)?;
    for _ in 0..args.span.duration / WINDOW_TICKS {
        let window_start = city.tick;
        let started = Instant::now();
        let stats = engine.run_window(&mut city, WINDOW_TICKS, args.check).map_err(fail)?;
        let window_ms = started.elapsed().as_secs_f64() * 1e3;
        sim_ms += window_ms;
        totals.add(stats);

        let frame = engine.harvest(&mut city, window_start).map_err(fail)?;
        if args.pipeline {
            writer.send(frame);
        } else {
            writer.send_and_wait(frame);
        }
        let census = engine.census(&city).map_err(fail)?;
        check_conservation(&census).map_err(fail)?;
        let checksum = engine.state_checksum(&city).map_err(fail)?;
        let imbalance = (threads > 1).then(|| (stats.imbalance(threads), stats.call_imbalance(threads)));
        let line = window_line(city.tick, &census, census.crossed - last.crossed, window_ms, imbalance, checksum);
        last = census;
        eprintln!("{line}");
        if let Some(log) = log.as_mut() {
            writeln!(log, "{line}").map_err(|e| fail(format!("window log: {e}")))?;
        }
    }

    let sensors = writer.finish().map_err(|e| fail(format!("sensor output: {e}")))?;
    let state_checksum = engine.state_checksum(&city).map_err(fail)?;
    if let Some(path) = &args.save_state {
        engine.download(&mut city).map_err(fail)?;
        save_snapshot(&city, path)?;
    }
    let balanced = args.backend == BackendKind::Balanced;
    let cost = format!("{:?}", args.engine.cost).to_lowercase();
    let report = RunReport {
        params: &city.params,
        backend: engine.name(),
        placement: engine.describe(),
        threads,
        block: args.engine.block,
        cost: if balanced { &cost } else { "-" },
        rebalance: if balanced { args.engine.rebalance_every } else { 0 },
        ticks: args.span.duration,
        census: last,
        wall_ms: sim_ms,
        phase_a_ms: totals.a_ns as f64 / 1e6,
        phase_b_ms: totals.b_ns as f64 / 1e6,
        sync_ms: totals.sync_ns() as f64 / 1e6,
        imbalance: totals.imbalance(threads),
        call_imbalance: totals.call_imbalance(threads),
        migration: totals.migration(),
        repartitions: totals.repartitions,
        compile_ms: engine.compile_ms(),
        write_ms: sensors.write_ms,
        sensor_bytes: sensors.bytes,
        state_checksum,
        sensor_checksum: sensors.checksum,
    };
    print!("{}", report.summary());
    if args.csv {
        println!("{}", report.csv_line());
    }
    Ok(())
}

fn compare_backends(args: CompareArgs) -> Result<(), Failure> {
    let mut a = make_engine(args.a, &args.engine).map_err(fail)?;
    let mut b = make_engine(args.b, &args.engine).map_err(fail)?;
    let mut city_a = make_city(&args.city, &args.span, true)?;
    let mut city_b = make_city(&args.city, &args.span, true)?;
    let p = &city_a.params;
    eprintln!(
        "[citysim] compare {} on {} with {} on {}, {}x{} city, {} to {}, checking every {}",
        a.name(),
        a.describe(),
        b.name(),
        b.describe(),
        p.rows,
        p.cols,
        stamp(city_a.tick),
        &stamp(city_a.tick + args.span.duration)[11..],
        count_of(args.every as usize, "tick")
    );
    let outcome =
        compare(a.as_mut(), &mut city_a, b.as_mut(), &mut city_b, args.span.duration, args.every).map_err(fail)?;
    match outcome {
        Ok(agreement) => {
            println!(
                "identical   {} checks over {} ticks, final state {:016x}",
                thousands(u64::from(agreement.checks)),
                thousands(u64::from(agreement.ticks)),
                agreement.state
            );
            Ok(())
        }
        Err(divergence) => {
            println!(
                "different   at {} (tick {}): {} state {:016x}, {} state {:016x}",
                stamp(divergence.tick),
                divergence.tick,
                a.name(),
                divergence.a,
                b.name(),
                divergence.b
            );
            if let Some(n) = divergence.node {
                let cols = city_a.params.cols as usize;
                println!("first node  {n} (row {}, col {})", n / cols, n % cols);
                println!("  {}: {}", a.name(), describe_node(&city_a, n));
                println!("  {}: {}", b.name(), describe_node(&city_b, n));
            }
            Err((String::new(), 1))
        }
    }
}

/// Memory the kernel reports as available, in bytes, when it can be read.
fn mem_available() -> Option<u64> {
    let text = std::fs::read_to_string("/proc/meminfo").ok()?;
    let line = text.lines().find(|l| l.starts_with("MemAvailable:"))?;
    let kib: u64 = line.split_whitespace().nth(1)?.parse().ok()?;
    Some(kib * 1024)
}

fn info(args: InfoArgs) {
    let (rows, cols) = args.city;
    let nodes = u64::from(rows) * u64::from(cols);
    let per_node = bytes_per_node(args.link_cells);
    let total = nodes * per_node;
    println!(
        "city        {rows} x {cols}: {} intersections, {} links, {} cells",
        thousands(nodes),
        thousands(4 * nodes),
        thousands(4 * nodes * u64::from(args.link_cells))
    );
    println!("memory      {} B per intersection, {:.3} GiB in total", thousands(per_node), gib(total));
    match mem_available() {
        Some(avail) => println!(
            "available   {:.2} GiB, so the city {}",
            gib(avail),
            if total < avail { "fits" } else { "does not fit" }
        ),
        None => println!("available   unknown"),
    }
}

fn selftest() -> Result<(), Failure> {
    let outputs = philox::known_answer_inputs().map(|(ctr, key)| philox::philox4x32_10(ctr, key));
    let passed = philox::known_answers_matched(&outputs);
    println!("philox      {passed} of {} Random123 known-answer vectors match on the CPU", philox::KNOWN_ANSWERS.len());
    if passed != philox::KNOWN_ANSWERS.len() {
        return Err(("Philox-4x32-10 does not match Random123 on the CPU".into(), 1));
    }
    gpu_selftest()
}

#[cfg(not(feature = "cuda"))]
fn gpu_selftest() -> Result<(), Failure> {
    println!("gpu         not built in; build with --features cuda to test the GPU");
    Ok(())
}

/// Opens the GPU, compiles the kernels, checks the device Philox against
/// Random123 and against the CPU, and times a kernel launch.
#[cfg(feature = "cuda")]
fn gpu_selftest() -> Result<(), Failure> {
    let fail = |e: String| (e, 1);
    let gpu = gpu::Gpu::open().map_err(fail)?;
    let info = &gpu.info;
    println!(
        "gpu         {}, compute {}.{}, {} MiB, driver CUDA {}.{}, NVRTC {}.{}",
        info.name,
        info.compute.0,
        info.compute.1,
        thousands(info.memory as u64 >> 20),
        info.driver.0,
        info.driver.1,
        info.nvrtc.0,
        info.nvrtc.1
    );
    println!(
        "kernels     compiled to an {} cubin in {:.1} ms, loaded in {:.2} ms",
        gpu.arch, gpu.compile_ms, gpu.load_ms
    );

    let passed = philox::known_answers_matched(&gpu.philox(&philox::known_answer_inputs()).map_err(fail)?);
    println!("philox      {passed} of {} Random123 known-answer vectors match on the GPU", philox::KNOWN_ANSWERS.len());
    if passed != philox::KNOWN_ANSWERS.len() {
        return Err(("Philox-4x32-10 does not match Random123 on the GPU".into(), 1));
    }

    let key = philox::key_from_seed(42);
    let inputs: Vec<_> =
        (0..SELFTEST_COUNTERS).map(|i| ([i, i ^ 0x5555_5555, 42, philox::STREAM_BRAKE], key)).collect();
    let device = gpu.philox(&inputs).map_err(fail)?;
    let differ =
        inputs.iter().zip(&device).filter(|((ctr, key), got)| philox::philox4x32_10(*ctr, *key) != **got).count();
    println!("philox      {} counters: GPU and CPU differ on {differ}", thousands(u64::from(SELFTEST_COUNTERS)));
    if differ > 0 {
        return Err(("the GPU and the CPU compute different Philox outputs".into(), 1));
    }

    let latency = gpu.launch_latency(SELFTEST_LAUNCHES).map_err(fail)?;
    println!(
        "launch      {:.1} us per empty kernel, mean of {} launches",
        latency.as_secs_f64() * 1e6,
        thousands(u64::from(SELFTEST_LAUNCHES))
    );
    Ok(())
}

fn main() -> ExitCode {
    let result = match Cli::parse().command {
        Command::Run(args) => run(args),
        Command::Compare(args) => compare_backends(args),
        Command::Info(args) => {
            info(args);
            Ok(())
        }
        Command::Selftest => selftest(),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err((message, code)) => {
            if !message.is_empty() {
                eprintln!("citysim: {message}");
            }
            ExitCode::from(code)
        }
    }
}
