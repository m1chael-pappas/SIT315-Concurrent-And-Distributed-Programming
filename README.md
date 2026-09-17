# SIT315 - Concurrent and Distributed Programming

Deakin unit work, one folder per task.
Each folder builds standalone and has its own README with the build line and arguments.

## Folders

| Folder | Task | Covers |
|---|---|---|
| [`task-m1`](task-m1) | M1 | Arduino external, pin change and timer interrupts |
| [`m2s2p`](m2s2p) | M2 seminar 2 | Vector addition with `std::thread` |
| [`m2t1p`](m2t1p) | M2 task 1 | OpenMP reductions, matrix multiply with pthreads and OpenMP |
| [`m2t2c`](m2t2c) | M2 task 2 | Parallel quicksort, OpenMP tasks and `std::thread` |
| [`m2t3d`](m2t3d) | M2 task 3 | Producer-consumer over a bounded buffer |
| [`m3s2p`](m3s2p) | M3 seminar 2 | MPI point-to-point and collectives |
| [`m3s3p`](m3s3p) | M3 seminar 3 | OpenCL kernels on GPU |
| [`m3t1p`](m3t1p) | M3 task 1 | Matrix multiply with MPI, hybrid MPI and OpenMP, hybrid MPI and OpenCL |

## Dependencies

```
sudo apt install g++ libomp-dev openmpi-bin libopenmpi-dev opencl-headers ocl-icd-opencl-dev
```

`m3s3p` and `m3t1p` also need an OpenCL runtime for your GPU vendor, or `pocl-opencl-icd` to run on the CPU.

The MPI folders are the exception to that apt line. Both the system OpenMPI and the system MPICH hang in `MPI_Init` on the WSL2 machine this was developed on, so `m3t1p` builds and runs against the MPICH that ships with Anaconda, by absolute path. `m3t1p/README.md` has the detail and the check for it.

## Building

There is no top-level Makefile.
Compile lines differ per folder and are documented in each README.

## Generated files

Datasets, compiled binaries and bulk program output are gitignored, so a clone is about 800 KB rather than 5.9 GB.
Each folder README gives the command to regenerate its own data.
All generators are seeded, so regenerated data matches what produced the committed results.
