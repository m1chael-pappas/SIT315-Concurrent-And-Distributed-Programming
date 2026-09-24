//! Command line: subcommands, flags, and parsing them into integer parameters.

use std::path::PathBuf;

use clap::{Args, Parser, Subcommand, ValueEnum};

use citysim::params::{Demand, Event, Params, WINDOW_TICKS, threshold};
use citysim::sched::partitioned::Cost;

/// Deterministic parallel city traffic simulator (SIT315 M4.T1D).
#[derive(Parser, Debug)]
#[command(name = "citysim", version)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

/// What to do.
#[derive(Subcommand, Debug)]
pub enum Command {
    /// Simulate a city and report timings, checksums and sensor output.
    Run(RunArgs),
    /// Run two backends side by side and stop at the first tick where their states differ.
    Compare(CompareArgs),
    /// Print the size and memory needs of a city without simulating it.
    Info(InfoArgs),
    /// Check the random number generator (and the GPU when built with --features cuda).
    Selftest,
}

/// How nodes are scheduled onto threads.
#[derive(ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
pub enum BackendKind {
    Seq,
    Static,
    Rayon,
    Balanced,
    Cuda,
}

/// The city: its size and the traffic model.
#[derive(Args, Debug, Clone)]
pub struct CityArgs {
    /// Grid of intersections, rows x columns, for example 128x128.
    #[arg(long, value_parser = parse_city)]
    pub city: (u32, u32),
    /// Cells per approach link; one cell is 7.5 m.
    #[arg(long, default_value_t = 32, value_parser = clap::value_parser!(u32).range(6..=250))]
    pub link_cells: u32,
    /// Random seed.
    #[arg(long, default_value_t = 42)]
    pub seed: u64,
    /// Nagel-Schreckenberg random braking probability.
    #[arg(long, default_value_t = 0.25)]
    pub brake: f64,
    /// Turn split in percent: straight, left, right.
    #[arg(long, default_value = "70,15,15", value_parser = parse_turns)]
    pub turns: (u32, u32, u32),
    /// Probability a car ends its trip (parks) on each link it enters.
    #[arg(long, default_value_t = 0.05)]
    pub trip_end: f64,
    /// Minimum green time in seconds.
    #[arg(long, default_value_t = 10)]
    pub min_green: u32,
    /// Maximum green time in seconds while the other road is waiting.
    #[arg(long, default_value_t = 45)]
    pub max_green: u32,
    /// Detector length in cells before the stop line.
    #[arg(long, default_value_t = 8)]
    pub detector: u32,
    /// Spawn probability per link per second downtown at peak hour.
    #[arg(long, default_value_t = 0.02)]
    pub peak: f64,
    /// Spawn probability per link per second far from downtown at peak hour.
    #[arg(long, default_value_t = 0.001)]
    pub floor: f64,
    /// Extra spawn probability on links entering from the city edge at peak hour.
    #[arg(long, default_value_t = 0.01)]
    pub edge: f64,
    /// Radius of the downtown peak in intersections; 0 means a sixth of the city.
    #[arg(long, default_value_t = 0)]
    pub spread: u32,
    /// Multiplier on all demand.
    #[arg(long, default_value_t = 1.0)]
    pub demand: f64,
    /// Event hotspot: ROW,COL,HH:MM,MINUTES[,PEAK]. PEAK defaults to 0.05.
    #[arg(long, value_parser = parse_event)]
    pub event: Option<(u32, u32, u32, u32, f64)>,
}

/// How a backend schedules its work.
#[derive(Args, Debug)]
pub struct EngineArgs {
    /// Worker threads for static, rayon and balanced; defaults to every logical CPU.
    #[arg(long)]
    pub threads: Option<usize>,
    /// Nodes per scheduling block; for cuda, threads per CUDA block.
    #[arg(long, default_value_t = 64)]
    pub block: usize,
    /// Block cost the balanced backend cuts by.
    #[arg(long, value_enum, default_value_t = Cost::Time)]
    pub cost: Cost,
    /// Windows between repartitions for balanced; 0 keeps the first partition.
    #[arg(long, default_value_t = 1)]
    pub rebalance_every: u32,
}

/// Where the simulation starts and how long it runs.
#[derive(Args, Debug)]
pub struct SpanArgs {
    /// Simulated start time of an empty city, HH:MM on a 5-minute boundary.
    #[arg(long, default_value = "06:00", value_parser = parse_clock, conflicts_with = "load_state")]
    pub start: u32,
    /// Simulated duration, such as 90m, 3h or 600s, a multiple of 5 minutes.
    #[arg(long, default_value = "1h", value_parser = parse_duration)]
    pub duration: u32,
    /// Start from this snapshot instead of an empty city.
    #[arg(long)]
    pub load_state: Option<PathBuf>,
}

/// Flags of `citysim run`.
#[derive(Args, Debug)]
pub struct RunArgs {
    #[command(flatten)]
    pub city: CityArgs,
    /// Scheduling backend.
    #[arg(long, value_enum, default_value_t = BackendKind::Seq)]
    pub backend: BackendKind,
    #[command(flatten)]
    pub engine: EngineArgs,
    #[command(flatten)]
    pub span: SpanArgs,
    /// Write a snapshot of the final state to this file.
    #[arg(long)]
    pub save_state: Option<PathBuf>,
    /// Write sensor counts to this CSV file.
    #[arg(long)]
    pub sensors: Option<PathBuf>,
    /// on: the writer thread formats a window while the next one runs; off: the run waits for it.
    #[arg(long, default_value = "on", value_parser = parse_switch)]
    pub pipeline: bool,
    /// Write the per-window lines to this file as well.
    #[arg(long)]
    pub window_log: Option<PathBuf>,
    /// Check every invariant after every tick.
    #[arg(long)]
    pub check: bool,
    /// Print the machine-readable CSV summary line.
    #[arg(long)]
    pub csv: bool,
}

/// Flags of `citysim compare`.
#[derive(Args, Debug)]
pub struct CompareArgs {
    #[command(flatten)]
    pub city: CityArgs,
    /// First backend.
    #[arg(long, value_enum, default_value_t = BackendKind::Seq)]
    pub a: BackendKind,
    /// Second backend.
    #[arg(long, value_enum)]
    pub b: BackendKind,
    #[command(flatten)]
    pub engine: EngineArgs,
    #[command(flatten)]
    pub span: SpanArgs,
    /// Ticks between state checksum comparisons.
    #[arg(long, default_value_t = 1, value_parser = clap::value_parser!(u32).range(1..))]
    pub every: u32,
}

/// Flags of `citysim info`.
#[derive(Args, Debug)]
pub struct InfoArgs {
    /// Grid of intersections, rows x columns.
    #[arg(long, value_parser = parse_city)]
    pub city: (u32, u32),
    /// Cells per approach link.
    #[arg(long, default_value_t = 32)]
    pub link_cells: u32,
}

fn parse_city(s: &str) -> Result<(u32, u32), String> {
    let (r, c) = s.split_once(['x', 'X']).ok_or("expected ROWSxCOLS, for example 128x128")?;
    let rows: u32 = r.trim().parse().map_err(|_| format!("bad row count: {r}"))?;
    let cols: u32 = c.trim().parse().map_err(|_| format!("bad column count: {c}"))?;
    if rows == 0 || cols == 0 {
        return Err("a city needs at least one row and one column".into());
    }
    Ok((rows, cols))
}

fn parse_clock(s: &str) -> Result<u32, String> {
    let (h, m) = s.split_once(':').ok_or("expected HH:MM")?;
    let h: u32 = h.parse().map_err(|_| format!("bad hour: {h}"))?;
    let m: u32 = m.parse().map_err(|_| format!("bad minute: {m}"))?;
    if h > 23 || m > 59 || m % 5 != 0 {
        return Err("time must be HH:MM on a 5-minute boundary".into());
    }
    Ok(h * 3600 + m * 60)
}

fn parse_duration(s: &str) -> Result<u32, String> {
    let (digits, unit) = s.split_at(s.find(|c: char| !c.is_ascii_digit()).unwrap_or(s.len()));
    let n: u32 = digits.parse().map_err(|_| format!("bad duration: {s}"))?;
    let seconds = match unit {
        "h" => n * 3600,
        "m" | "min" => n * 60,
        "s" | "" => n,
        _ => return Err(format!("unknown duration unit in {s}; use h, m or s")),
    };
    if seconds == 0 || seconds % WINDOW_TICKS != 0 {
        return Err("duration must be a positive multiple of 5 minutes".into());
    }
    Ok(seconds)
}

fn parse_switch(s: &str) -> Result<bool, String> {
    match s {
        "on" => Ok(true),
        "off" => Ok(false),
        _ => Err("expected on or off".into()),
    }
}

fn parse_turns(s: &str) -> Result<(u32, u32, u32), String> {
    let parts: Vec<u32> =
        s.split(',').map(|p| p.trim().parse().map_err(|_| format!("bad turn share: {p}"))).collect::<Result<_, _>>()?;
    match parts[..] {
        [a, b, c] if a + b + c == 100 => Ok((a, b, c)),
        [_, _, _] => Err("turn shares must add up to 100".into()),
        _ => Err("expected three shares: straight,left,right".into()),
    }
}

fn parse_event(s: &str) -> Result<(u32, u32, u32, u32, f64), String> {
    let parts: Vec<&str> = s.split(',').collect();
    if !(4..=5).contains(&parts.len()) {
        return Err("expected ROW,COL,HH:MM,MINUTES[,PEAK]".into());
    }
    let row = parts[0].parse().map_err(|_| "bad event row")?;
    let col = parts[1].parse().map_err(|_| "bad event column")?;
    let at = parse_clock(parts[2])?;
    let minutes = parts[3].parse().map_err(|_| "bad event minutes")?;
    let peak = parts.get(4).map_or(Ok(0.05), |p| p.parse().map_err(|_| "bad event peak"))?;
    Ok((row, col, at, minutes, peak))
}

impl CityArgs {
    /// Integer parameters for a run starting at `start_tick`.
    pub fn params(&self, start_tick: u32) -> Params {
        let (straight, left, _) = self.turns;
        Params {
            rows: self.city.0,
            cols: self.city.1,
            link_cells: self.link_cells,
            seed: self.seed,
            brake: threshold(self.brake),
            turn_straight: threshold(f64::from(straight) / 100.0),
            turn_straight_left: threshold(f64::from(straight + left) / 100.0),
            trip_end: threshold(self.trip_end),
            min_green: self.min_green,
            max_green: self.max_green.max(self.min_green),
            detector: self.detector.min(self.link_cells),
            start_tick,
        }
    }

    /// The demand model, scaled by `--demand`.
    pub fn demand(&self) -> Demand {
        let scale = |p: f64| threshold(p * self.demand);
        let spread = if self.spread > 0 { self.spread } else { (self.city.0.max(self.city.1) / 6).max(1) };
        let event = self.event.map(|(row, col, at, minutes, peak)| Event {
            row,
            col,
            start_tick: at,
            end_tick: at + minutes * 60,
            peak: scale(peak),
            spread: (spread / 3).max(1),
        });
        Demand { peak: scale(self.peak), floor: scale(self.floor), spread, edge: scale(self.edge), event }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_city_sizes() {
        assert_eq!(parse_city("128x64"), Ok((128, 64)));
        assert!(parse_city("0x5").is_err());
        assert!(parse_city("12").is_err());
    }

    #[test]
    fn parses_times_and_durations() {
        assert_eq!(parse_clock("07:30"), Ok(27_000));
        assert!(parse_clock("07:31").is_err());
        assert_eq!(parse_duration("3h"), Ok(10_800));
        assert_eq!(parse_duration("90m"), Ok(5_400));
        assert!(parse_duration("7m").is_err());
    }

    #[test]
    fn turn_shares_must_sum_to_100() {
        assert_eq!(parse_turns("70,15,15"), Ok((70, 15, 15)));
        assert!(parse_turns("70,20,15").is_err());
    }

    #[test]
    fn parses_events() {
        assert_eq!(parse_event("10,12,08:00,30"), Ok((10, 12, 28_800, 30, 0.05)));
        assert_eq!(parse_event("1,2,17:30,15,0.1"), Ok((1, 2, 63_000, 15, 0.1)));
    }
}
