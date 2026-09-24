//! GPU device set-up, runtime-compiled kernels and the GPU self-test.
//!
//! `kernels.cu` is compiled at start-up with `prelude()` in front of it, by
//! NVRTC, to a cubin for the architecture of device 0. This is the only module
//! allowed `unsafe`, for raw NVRTC calls and kernel launches.

use std::ffi::{CString, c_char, c_int};
use std::sync::Arc;
use std::time::{Duration, Instant};

use cudarc::driver::{CudaContext, CudaFunction, CudaModule, CudaStream, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::{Ptx, result, sys};

use crate::philox;

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

/// The `#define`s `kernels.cu` expects, generated from the Rust constants.
pub fn prelude() -> String {
    let constants = [
        ("PHILOX_M0", philox::M0),
        ("PHILOX_M1", philox::M1),
        ("PHILOX_W0", philox::W0),
        ("PHILOX_W1", philox::W1),
        ("PHILOX_ROUNDS", philox::ROUNDS),
        ("STREAM_BRAKE", philox::STREAM_BRAKE),
        ("STREAM_ROUTE", philox::STREAM_ROUTE),
        ("STREAM_SPAWN", philox::STREAM_SPAWN),
    ];
    constants.iter().map(|(name, value)| format!("#define {name} {value:#x}u\n")).collect()
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prelude_defines_every_philox_constant() {
        let prelude = prelude();
        assert!(prelude.contains("#define PHILOX_M0 0xd2511f53u\n"));
        assert!(prelude.contains("#define PHILOX_ROUNDS 0xau\n"));
        assert_eq!(prelude.lines().count(), 8);
    }
}
