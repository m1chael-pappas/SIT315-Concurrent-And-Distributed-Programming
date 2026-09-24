//! Sensor output in the m2t3d/m3t3d CSV format, written on its own thread.
//!
//! Every traffic light reports the cars that crossed its stop line in each
//! 5-minute window, one row per light per window, in time order and then light
//! order, with no header:
//!
//! ```text
//! 2026-08-01 08:00,TL0042,37
//! ```
//!
//! The simulation hands each finished window to a writer thread through a
//! bounded channel, so formatting and disk writes overlap with the next window.
//! A frame is any `Frame`; the CPU backends pass a `VecFrame`.

use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::PathBuf;
use std::sync::mpsc::{Receiver, Sender, SyncSender, channel, sync_channel};
use std::thread::JoinHandle;
use std::time::Instant;

use crate::checksum::SensorChecksum;

/// Days from 1970-01-01 to a civil date, Howard Hinnant's algorithm as in m2t3d's `daysFromCivil`.
pub fn days_from_civil(y: i64, m: u32, d: u32) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let mp = i64::from(if m > 2 { m - 3 } else { m + 9 });
    let doy = (153 * mp + 2) / 5 + i64::from(d) - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

/// Civil date of a day count, the inverse of `days_from_civil`, as in m2t3d's `civilFromDays`.
pub fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (if m <= 2 { y + 1 } else { y }, m, d)
}

/// Ticks count seconds from this date, 2026-08-01 00:00, the base date of the m2t3d data.
pub fn base_days() -> i64 {
    days_from_civil(2026, 8, 1)
}

/// "YYYY-MM-DD HH:MM" for an absolute tick.
pub fn stamp(tick: u32) -> String {
    let days = base_days() + i64::from(tick / 86_400);
    let (y, m, d) = civil_from_days(days);
    let hh = (tick % 86_400) / 3600;
    let mm = (tick % 3600) / 60;
    format!("{y:04}-{m:02}-{d:02} {hh:02}:{mm:02}")
}

fn append_uint(buf: &mut Vec<u8>, mut v: u64) {
    let mut digits = [0u8; 20];
    let mut n = 0;
    loop {
        digits[n] = b'0' + (v % 10) as u8;
        n += 1;
        v /= 10;
        if v == 0 {
            break;
        }
    }
    buf.extend(digits[..n].iter().rev());
}

/// Appends one window's rows: one per light, lights in id order, ids zero-padded to four digits.
pub fn format_window(buf: &mut Vec<u8>, window_start: u32, counts: &[u32]) {
    let stamp = stamp(window_start);
    for (light, &count) in counts.iter().enumerate() {
        buf.extend_from_slice(stamp.as_bytes());
        buf.extend_from_slice(b",TL");
        for pad in [1000, 100, 10] {
            if light < pad {
                buf.push(b'0');
            }
        }
        append_uint(buf, light as u64);
        buf.push(b',');
        append_uint(buf, u64::from(count));
        buf.push(b'\n');
    }
}

/// A finished sensor window: when it started and one count per light.
pub trait Frame: Send + 'static {
    fn window_start(&self) -> u32;
    fn counts(&self) -> &[u32];
}

/// A frame held in ordinary host memory, used by the CPU backends.
pub struct VecFrame {
    pub window_start: u32,
    pub counts: Vec<u32>,
}

impl Frame for VecFrame {
    fn window_start(&self) -> u32 {
        self.window_start
    }

    fn counts(&self) -> &[u32] {
        &self.counts
    }
}

/// What the writer thread did over the whole run.
pub struct SensorReport {
    pub checksum: u64,
    pub bytes: u64,
    pub write_ms: f64,
}

/// Handle to the writer thread.
pub struct SensorWriter {
    tx: Option<SyncSender<Box<dyn Frame>>>,
    done: Receiver<()>,
    sent: u64,
    confirmed: u64,
    handle: Option<JoinHandle<io::Result<SensorReport>>>,
}

fn writer_loop(
    frames: Receiver<Box<dyn Frame>>,
    done: Sender<()>,
    path: Option<PathBuf>,
    lights: usize,
) -> io::Result<SensorReport> {
    let mut out = path.map(File::create).transpose()?.map(|f| BufWriter::with_capacity(8 << 20, f));
    let mut sum = SensorChecksum::new(lights);
    let mut buf = Vec::with_capacity(lights * 28);
    let (mut bytes, mut write_ms) = (0u64, 0.0);
    for frame in frames {
        let start = Instant::now();
        sum.add_window(frame.window_start(), frame.counts());
        if let Some(out) = out.as_mut() {
            buf.clear();
            format_window(&mut buf, frame.window_start(), frame.counts());
            out.write_all(&buf)?;
            bytes += buf.len() as u64;
        }
        write_ms += start.elapsed().as_secs_f64() * 1e3;
        drop(frame);
        let _ = done.send(());
    }
    if let Some(mut out) = out {
        out.flush()?;
    }
    Ok(SensorReport { checksum: sum.finish(), bytes, write_ms })
}

impl SensorWriter {
    /// Starts the writer thread. Without a path it still computes the sensor checksum.
    /// `depth` bounds how many frames can wait, which is the back-pressure on the simulation.
    pub fn start(path: Option<PathBuf>, lights: usize, depth: usize) -> SensorWriter {
        let (tx, rx) = sync_channel(depth);
        let (done_tx, done) = channel();
        let handle = std::thread::spawn(move || writer_loop(rx, done_tx, path, lights));
        SensorWriter { tx: Some(tx), done, sent: 0, confirmed: 0, handle: Some(handle) }
    }

    /// Hands a finished window to the writer, blocking while `depth` frames wait.
    /// A frame sent after the writer failed is dropped; `finish` returns the failure.
    pub fn send(&mut self, frame: Box<dyn Frame>) {
        if let Some(tx) = &self.tx
            && tx.send(frame).is_ok()
        {
            self.sent += 1;
        }
    }

    /// `send`, then blocks until the writer has written and dropped every frame sent so far.
    pub fn send_and_wait(&mut self, frame: Box<dyn Frame>) {
        self.send(frame);
        while self.confirmed < self.sent && self.done.recv().is_ok() {
            self.confirmed += 1;
        }
    }

    /// Closes the channel, waits for the writer and returns its report.
    pub fn finish(mut self) -> io::Result<SensorReport> {
        drop(self.tx.take());
        match self.handle.take().map(JoinHandle::join) {
            Some(Ok(report)) => report,
            _ => Err(io::Error::other("sensor writer thread panicked")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base_date_matches_m2t3d() {
        assert_eq!(stamp(0), "2026-08-01 00:00");
        assert_eq!(stamp(8 * 3600 + 5 * 60), "2026-08-01 08:05");
    }

    #[test]
    fn dates_roll_over_months_and_years() {
        assert_eq!(stamp(31 * 86_400), "2026-09-01 00:00");
        assert_eq!(stamp(153 * 86_400), "2027-01-01 00:00");
        assert_eq!(civil_from_days(days_from_civil(2028, 2, 29)), (2028, 2, 29));
        assert_eq!(civil_from_days(days_from_civil(2028, 3, 1) - 1), (2028, 2, 29));
    }

    #[test]
    fn rows_match_the_m2t3d_format() {
        let mut buf = Vec::new();
        format_window(&mut buf, 8 * 3600, &[24, 0, 7]);
        assert_eq!(
            String::from_utf8(buf).unwrap(),
            "2026-08-01 08:00,TL0000,24\n2026-08-01 08:00,TL0001,0\n2026-08-01 08:00,TL0002,7\n"
        );
    }

    #[test]
    fn wide_ids_are_not_truncated() {
        let mut buf = Vec::new();
        let counts = vec![0; 10_001];
        format_window(&mut buf, 0, &counts);
        assert!(String::from_utf8(buf).unwrap().ends_with("2026-08-01 00:00,TL10000,0\n"));
    }

    #[test]
    fn writer_thread_checksums_without_a_file() {
        let mut writer = SensorWriter::start(None, 5, 2);
        writer.send(Box::new(VecFrame { window_start: 8 * 3600, counts: vec![0, 60, 15, 120, 1] }));
        writer.send(Box::new(VecFrame { window_start: 9 * 3600, counts: vec![0, 50, 50, 0, 90] }));
        let report = writer.finish().unwrap();
        assert_eq!(report.checksum, 17_191_051_930_829_670_833);
        assert_eq!(report.bytes, 0);
    }
}
