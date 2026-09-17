// ============================================================================
// SIT315 Module 3, Task M3.T2C - Complex distributed computing
// Distributed sort with MPI + OpenCL.
//
// The MPI layer is identical to quicksort_mpi.cpp: the same parallel sorting by
// regular sampling, from the same psrs.h. The only difference is step 2. Where
// quicksort_mpi sorts a rank's chunk with the m2t2c quicksort on the CPU, this
// program uploads the chunk to an OpenCL device and sorts it there with a
// bitonic sorting network.
//
// Data movement per rank, per run:
//   host -> device   the padded chunk, once, at the start of the local sort
//   device           m(m+1)/2 kernel launches over that buffer, nothing copied
//   device -> host   the sorted chunk, once, at the end
// Everything after that (sampling, pivots, all-to-all, merge, gather) is MPI on
// the host and never touches the device.
//
// Build:  MPICH_CXX=g++ mpic++ -O2 -std=c++17 -Wall -Wextra -o quicksort_mpi_ocl quicksort_mpi_ocl.cpp -lOpenCL
// Run:    mpirun -np 4 ./quicksort_mpi_ocl --n 16000000 --runs 5
//
// bitonic.cl is read from the working directory at run time.
// ============================================================================

#define CL_TARGET_OPENCL_VERSION 120

#include "psrs.h"
#include <CL/cl.h>
#include <climits>
#include <cstring>

static const char* KERNEL_FILE = "./bitonic.cl";

static void die(const char* what, cl_int err) {
    std::fprintf(stderr, "%s failed with OpenCL error %d\n", what, err);
    MPI_Abort(MPI_COMM_WORLD, 1);
}

// Holds one rank's OpenCL device, context, queue and kernel for the whole run.
// Built once, outside the timed section, because compiling the program from
// source costs the same whatever n is.
struct Device {
    cl_device_id id = nullptr;
    cl_context ctx = nullptr;
    cl_command_queue queue = nullptr;
    cl_program program = nullptr;
    cl_kernel kernel = nullptr;
    char name[256] = {0};
    char vendor[256] = {0};
    cl_device_type type = 0;
    cl_uint computeUnits = 0;
    size_t maxWorkGroup = 0;

    const char* typeName() const {
        if (type & CL_DEVICE_TYPE_GPU) return "GPU";
        if (type & CL_DEVICE_TYPE_CPU) return "CPU";
        return "other";
    }
};

static void pickDevice(Device& d) {
    cl_platform_id platform;
    cl_uint count = 0;
    cl_int err = clGetPlatformIDs(1, &platform, &count);
    if (err < 0 || count == 0) die("clGetPlatformIDs", err);

    // GPU first, CPU as the fallback. Which one it lands on gets printed.
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &d.id, NULL);
    if (err != CL_SUCCESS) {
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &d.id, NULL);
        if (err < 0) die("clGetDeviceIDs", err);
    }

    clGetDeviceInfo(d.id, CL_DEVICE_NAME, sizeof d.name, d.name, NULL);
    clGetDeviceInfo(d.id, CL_DEVICE_VENDOR, sizeof d.vendor, d.vendor, NULL);
    clGetDeviceInfo(d.id, CL_DEVICE_TYPE, sizeof d.type, &d.type, NULL);
    clGetDeviceInfo(d.id, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof d.computeUnits, &d.computeUnits, NULL);
    clGetDeviceInfo(d.id, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof d.maxWorkGroup, &d.maxWorkGroup, NULL);
}

static void buildKernel(Device& d) {
    cl_int err;
    d.ctx = clCreateContext(NULL, 1, &d.id, NULL, NULL, &err);
    if (err < 0) die("clCreateContext", err);
    d.queue = clCreateCommandQueue(d.ctx, d.id, 0, &err);
    if (err < 0) die("clCreateCommandQueue", err);

    std::FILE* f = std::fopen(KERNEL_FILE, "r");
    if (!f) {
        std::fprintf(stderr, "cannot open %s, run from the folder holding it\n", KERNEL_FILE);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    std::fseek(f, 0, SEEK_END);
    size_t size = static_cast<size_t>(std::ftell(f));
    std::rewind(f);
    std::string src(size, '\0');
    if (std::fread(&src[0], 1, size, f) != size) die("reading kernel source", 0);
    std::fclose(f);

    const char* text = src.c_str();
    d.program = clCreateProgramWithSource(d.ctx, 1, &text, &size, &err);
    if (err < 0) die("clCreateProgramWithSource", err);

    err = clBuildProgram(d.program, 0, NULL, NULL, NULL, NULL);
    if (err < 0) {
        size_t logSize = 0;
        clGetProgramBuildInfo(d.program, d.id, CL_PROGRAM_BUILD_LOG, 0, NULL, &logSize);
        std::string log(logSize, '\0');
        clGetProgramBuildInfo(d.program, d.id, CL_PROGRAM_BUILD_LOG, logSize, &log[0], NULL);
        std::fprintf(stderr, "kernel build log:\n%s\n", log.c_str());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    d.kernel = clCreateKernel(d.program, "bitonic_step", &err);
    if (err < 0) die("clCreateKernel", err);
}

static size_t nextPowerOfTwo(size_t v) {
    size_t p = 1;
    while (p < v) p <<= 1;
    return p;
}

// Sorts count elements of `a` on the device with a bitonic network.
// localSize of 0 means pass NULL and let the runtime choose the work-group size.
static void bitonicSort(Device& d, int32_t* a, int64_t count, size_t localSize,
                        long& launches, double& uploadMs, double& kernelMs, double& readMs) {
    if (count <= 1) return;

    size_t padded = nextPowerOfTwo(static_cast<size_t>(count));
    if (localSize > 0) {
        // The global size must be a whole number of work-groups.
        while (localSize > padded || (padded % localSize) != 0) localSize >>= 1;
        if (localSize == 0) localSize = 1;
    }

    std::vector<int32_t> buf(padded, INT32_MAX);
    std::memcpy(buf.data(), a, static_cast<size_t>(count) * sizeof(int32_t));

    qs::Timer t;
    cl_int err;

    t.start();
    cl_mem dev = clCreateBuffer(d.ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                padded * sizeof(int32_t), buf.data(), &err);
    if (err < 0) die("clCreateBuffer", err);
    clFinish(d.queue);
    uploadMs += t.stop_ms();

    t.start();
    clSetKernelArg(d.kernel, 0, sizeof(cl_mem), &dev);
    const size_t global = padded;
    const size_t* localPtr = (localSize > 0) ? &localSize : NULL;

    for (cl_uint k = 2; k <= padded; k <<= 1) {
        for (cl_uint j = k >> 1; j > 0; j >>= 1) {
            clSetKernelArg(d.kernel, 1, sizeof(cl_uint), &j);
            clSetKernelArg(d.kernel, 2, sizeof(cl_uint), &k);
            err = clEnqueueNDRangeKernel(d.queue, d.kernel, 1, NULL, &global, localPtr,
                                         0, NULL, NULL);
            if (err < 0) die("clEnqueueNDRangeKernel", err);
            ++launches;
        }
    }
    clFinish(d.queue);
    kernelMs += t.stop_ms();

    t.start();
    err = clEnqueueReadBuffer(d.queue, dev, CL_TRUE, 0,
                              static_cast<size_t>(count) * sizeof(int32_t), a, 0, NULL, NULL);
    if (err < 0) die("clEnqueueReadBuffer", err);
    readMs += t.stop_ms();

    clReleaseMemObject(dev);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, procs, nameLen;
    char host[MPI_MAX_PROCESSOR_NAME];
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &procs);
    MPI_Get_processor_name(host, &nameLen);

    const int64_t n       = qs::arg_ll(argc, argv, "--n", 16000000);
    const uint64_t seed   = static_cast<uint64_t>(qs::arg_ll(argc, argv, "--seed", 315));
    const int runs        = static_cast<int>(qs::arg_ll(argc, argv, "--runs", 5));
    const size_t localSz  = static_cast<size_t>(qs::arg_ll(argc, argv, "--local", 0));
    const std::string pat = qs::arg_str(argc, argv, "--pattern", "random");

    if (n <= 0 || runs <= 0) {
        if (rank == 0) std::fprintf(stderr, "--n and --runs must be positive\n");
        MPI_Finalize();
        return 1;
    }
    if (!psrs::usable(n, procs)) {
        if (rank == 0)
            std::fprintf(stderr, "need at least p*p elements to sample from: n=%lld, p=%d\n",
                         static_cast<long long>(n), procs);
        MPI_Finalize();
        return 1;
    }

    Device dev;
    pickDevice(dev);
    buildKernel(dev);

    // Every rank says which device it got, which is what proves the kernel ran
    // where the report claims it did.
    std::printf("rank %d on %s using %s device \"%s\" (%s, %u compute units, max work-group %zu)\n",
                rank, host, dev.typeName(), dev.name, dev.vendor,
                dev.computeUnits, dev.maxWorkGroup);
    std::fflush(stdout);
    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) {
        char lib[MPI_MAX_LIBRARY_VERSION_STRING] = {0};
        int len = 0;
        MPI_Get_library_version(lib, &len);
        for (int i = 0; i < len; ++i) if (lib[i] == '\n') { lib[i] = '\0'; break; }
        std::fprintf(stderr, "[mpi+ocl] n=%lld procs=%d runs=%d local=%s\n",
                     static_cast<long long>(n), procs, runs,
                     localSz ? std::to_string(localSz).c_str() : "runtime choice");
        std::fprintf(stderr, "      device: %s %s\n", dev.typeName(), dev.name);
        std::fprintf(stderr, "      linked against %s\n", lib);
        std::fflush(stderr);
    }

    std::vector<int32_t> original, work;
    uint64_t want = 0;
    if (rank == 0) {
        std::fprintf(stderr, "  generating %lld elements (pattern=%s seed=%llu)\n",
                     static_cast<long long>(n), pat.c_str(),
                     static_cast<unsigned long long>(seed));
        std::fflush(stderr);
        original = qs::make_dataset(n, seed, pat);
        want = qs::checksum(original.data(), original.size());
        work.resize(original.size());
        std::printf("program   quicksort_mpi_ocl\n");
        std::printf("elements  %lld\n", static_cast<long long>(n));
        std::printf("bytes     %lld\n", static_cast<long long>(n) * 4);
        std::printf("processes %d\n", procs);
        std::printf("device    %s %s\n", dev.typeName(), dev.name);
        std::printf("checksum  %llu\n", static_cast<unsigned long long>(want));
        std::printf("runs      %d\n\n", runs);
    }

    long launches = 0;
    double uploadMs = 0, kernelMs = 0, readMs = 0;
    auto localSort = [&](int32_t* a, int64_t count) {
        bitonicSort(dev, a, count, localSz, launches, uploadMs, kernelMs, readMs);
    };

    std::vector<double> times;
    psrs::Phases last;
    for (int r = 0; r < runs; ++r) {
        if (rank == 0) {
            std::copy(original.begin(), original.end(), work.begin());
            std::fprintf(stderr, "  run %d of %d\n", r + 1, runs);
            std::fflush(stderr);
        }
        launches = 0; uploadMs = kernelMs = readMs = 0;
        MPI_Barrier(MPI_COMM_WORLD);

        psrs::Phases ph;
        psrs::sort(work, n, rank, procs, localSort, ph);

        if (rank == 0) {
            bool ok = qs::is_sorted_asc(work.data(), work.size()) &&
                      qs::checksum(work.data(), work.size()) == want;
            std::printf("run %d    %10.2f ms   %s   (%ld launches, upload %.1f, kernel %.1f, read %.1f ms)\n",
                        r + 1, ph.total(), ok ? "verified" : "FAILED",
                        launches, uploadMs, kernelMs, readMs);
            if (!ok) { MPI_Abort(MPI_COMM_WORLD, 1); }
            times.push_back(ph.total());
            last = ph;
        }
    }

    if (rank == 0) {
        double med = qs::median_of(times);
        std::printf("\nmedian    %10.2f ms\n", med);
        std::printf("phases    scatter %.2f  localsort %.2f  pivots %.2f  "
                    "exchange %.2f  merge %.2f  gather %.2f  (last run, ms)\n",
                    last.scatter, last.localsort, last.pivots,
                    last.exchange, last.merge, last.gather);
        std::printf("kernel    %ld launches, global work size %lld per launch, local %s\n",
                    launches, static_cast<long long>(nextPowerOfTwo(
                        static_cast<size_t>((n + procs - 1) / procs))),
                    localSz ? std::to_string(localSz).c_str() : "runtime choice");
        std::printf("CSV,mpi+ocl,%lld,%d,bitonic,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%llu,%s %s\n",
                    static_cast<long long>(n), procs, med,
                    last.scatter, last.localsort, last.pivots,
                    last.exchange, last.merge, last.gather,
                    static_cast<unsigned long long>(want), dev.typeName(), dev.name);
    }

    clReleaseKernel(dev.kernel);
    clReleaseProgram(dev.program);
    clReleaseCommandQueue(dev.queue);
    clReleaseContext(dev.ctx);
    MPI_Finalize();
    return 0;
}
