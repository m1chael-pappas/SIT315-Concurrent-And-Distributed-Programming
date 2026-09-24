//! Run reporting: the per-window log, the summary and the machine-readable CSV line.

use crate::invariants::Census;
use crate::params::Params;
use crate::sensors::stamp;

/// Everything the summary needs about a finished run.
pub struct RunReport<'a> {
    pub params: &'a Params,
    pub backend: &'a str,
    pub placement: String,
    pub threads: usize,
    pub block: usize,
    pub cost: &'a str,
    pub rebalance: u32,
    pub ticks: u32,
    pub census: Census,
    pub wall_ms: f64,
    pub phase_a_ms: f64,
    pub phase_b_ms: f64,
    pub sync_ms: f64,
    pub imbalance: f64,
    pub call_imbalance: f64,
    pub migration: f64,
    pub repartitions: u32,
    pub compile_ms: f64,
    pub write_ms: f64,
    pub sensor_bytes: u64,
    pub state_checksum: u64,
    pub sensor_checksum: u64,
}

/// `n` and `noun`, with an `s` on the noun unless `n` is 1.
pub fn count_of(n: usize, noun: &str) -> String {
    format!("{n} {noun}{}", if n == 1 { "" } else { "s" })
}

/// `n` with a comma between each group of three digits.
pub fn thousands(n: u64) -> String {
    let digits = n.to_string();
    let mut out = String::with_capacity(digits.len() + digits.len() / 3);
    for (i, ch) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i) % 3 == 0 {
            out.push(',');
        }
        out.push(ch);
    }
    out
}

/// The line printed per window: time, cars on the road, share stopped, crossings,
/// wall time, per-phase and per-window load imbalance when given, and the state checksum.
pub fn window_line(
    tick: u32,
    census: &Census,
    crossed_in_window: u64,
    window_ms: f64,
    imbalance: Option<(f64, f64)>,
    checksum: u64,
) -> String {
    let stopped_pct = if census.active == 0 { 0.0 } else { 100.0 * census.stopped as f64 / census.active as f64 };
    let balance = imbalance.map_or(String::new(), |(phase, window)| format!("  imbalance {phase:.3} {window:.3}"));
    format!(
        "  {}  cars {:>10}  stopped {:>5.1}%  crossings {:>9}  {:>9.1} ms{balance}  state {:016x}",
        &stamp(tick)[11..],
        thousands(census.active),
        stopped_pct,
        thousands(crossed_in_window),
        window_ms,
        checksum
    )
}

impl RunReport<'_> {
    fn real_time_ratio(&self) -> f64 {
        if self.wall_ms > 0.0 { f64::from(self.ticks) * 1e3 / self.wall_ms } else { 0.0 }
    }

    /// The human-readable summary printed to stdout.
    pub fn summary(&self) -> String {
        let p = self.params;
        let c = &self.census;
        let conservation = if c.spawned == c.active + c.exited + c.parked { "ok" } else { "BROKEN" };
        let per_tick = if self.ticks > 0 { self.wall_ms / f64::from(self.ticks) } else { 0.0 };
        let mut s = String::new();
        s += "program     citysim\n";
        s += &format!(
            "city        {} x {}: {} intersections, {} links, {} cells of 7.5 m ({} m links)\n",
            p.rows,
            p.cols,
            thousands(p.nodes() as u64),
            thousands(p.links() as u64),
            thousands(p.links() as u64 * u64::from(p.link_cells)),
            p.link_cells as f64 * 7.5
        );
        s += &format!("backend     {} on {}\n", self.backend, self.placement);
        s += &format!(
            "simulated   {} to {}, {} ticks\n",
            stamp(p.start_tick),
            stamp(p.start_tick + self.ticks),
            thousands(u64::from(self.ticks))
        );
        s += &format!(
            "cars        spawned {}  exited {}  parked {}  on the road {}  conservation {}\n",
            thousands(c.spawned),
            thousands(c.exited),
            thousands(c.parked),
            thousands(c.active),
            conservation
        );
        s += &format!("crossings   {}\n", thousands(c.crossed));
        s += &format!(
            "time        wall {:.1} ms  per tick {:.3} ms  phase A {:.1} ms  phase B {:.1} ms  real time x{:.0}\n",
            self.wall_ms,
            per_tick,
            self.phase_a_ms,
            self.phase_b_ms,
            self.real_time_ratio()
        );
        if self.threads > 1 {
            s += &format!(
                "balance     imbalance per phase {:.3}, per window {:.3}  sync {:.1} ms  migration {:.1}%  repartitions {}\n",
                self.imbalance,
                self.call_imbalance,
                self.sync_ms,
                100.0 * self.migration,
                self.repartitions
            );
        }
        if self.compile_ms > 0.0 {
            s += &format!("kernels     compiled in {:.1} ms, outside the wall time\n", self.compile_ms);
        }
        if self.sensor_bytes > 0 {
            s += &format!(
                "sensors     {:.1} MiB written, writer thread busy {:.1} ms\n",
                self.sensor_bytes as f64 / (1024.0 * 1024.0),
                self.write_ms
            );
        }
        s += &format!("checksums   state {:016x}  sensor {}\n", self.state_checksum, self.sensor_checksum);
        s
    }

    /// One comma-separated line with every number, prefixed with CSV so scripts can grep for it.
    pub fn csv_line(&self) -> String {
        let p = self.params;
        let c = &self.census;
        format!(
            "CSV,citysim,{},{},{},{},{},{}x{},{},{},{},{},{},{},{:.3},{:.5},{:.1},{:.3},{:.3},{:.3},{:.4},{:.4},{:.4},{},{:.3},{:.3},{:016x},{}",
            self.backend,
            self.threads,
            self.block,
            self.cost,
            self.rebalance,
            p.rows,
            p.cols,
            p.link_cells,
            self.ticks,
            c.spawned,
            c.exited,
            c.parked,
            c.active,
            self.wall_ms,
            if self.ticks > 0 { self.wall_ms / f64::from(self.ticks) } else { 0.0 },
            self.real_time_ratio(),
            self.phase_a_ms,
            self.phase_b_ms,
            self.sync_ms,
            self.imbalance,
            self.call_imbalance,
            self.migration,
            self.repartitions,
            self.compile_ms,
            self.write_ms,
            self.state_checksum,
            self.sensor_checksum
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn thousands_separators() {
        assert_eq!(thousands(0), "0");
        assert_eq!(thousands(999), "999");
        assert_eq!(thousands(1_000), "1,000");
        assert_eq!(thousands(12_345_678), "12,345,678");
    }
}
