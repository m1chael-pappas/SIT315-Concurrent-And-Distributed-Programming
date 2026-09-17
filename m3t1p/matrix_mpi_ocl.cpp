// SIT315 M3.T1P - Hybrid MPI and OpenCL matrix multiplication.
//
// The MPI layer is identical to matrix_mpi.cpp: B broadcast whole, A scattered
// by row bands, C gathered back. The difference is inside a rank, where the
// band is handed to an OpenCL device instead of being walked by the CPU.
// Every rank builds its own context, queue and program, so each one drives its
// device independently and the ranks only meet again at the gather.
//
// Build:  MPICH_CXX=g++ mpic++ -O2 -o matrix_mpi_ocl matrix_mpi_ocl.cpp -lOpenCL
// Run:    mpirun -np 4 ./matrix_mpi_ocl 1000
//
// matmul.cl is read from the working directory at run time, so run from here.

#define CL_TARGET_OPENCL_VERSION 120

#include "matmul_common.h"
#include <CL/cl.h>
#include <cstring>
#include <string>

static const char* KERNEL_FILE = "./matmul.cl";

static void die(const char* what, cl_int err) {
    std::fprintf(stderr, "%s failed with OpenCL error %d\n", what, err);
    MPI_Abort(MPI_COMM_WORLD, 1);
}

// First platform, GPU if it has one, otherwise CPU.
static cl_device_id pickDevice(bool& isGpu) {
    cl_platform_id platform;
    cl_uint count = 0;
    cl_int err = clGetPlatformIDs(1, &platform, &count);
    if (err < 0 || count == 0) die("clGetPlatformIDs", err);

    cl_device_id device;
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    isGpu = (err == CL_SUCCESS);
    if (!isGpu) {
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &device, NULL);
        if (err < 0) die("clGetDeviceIDs", err);
    }
    return device;
}

static cl_program buildProgram(cl_context ctx, cl_device_id dev) {
    std::FILE* f = std::fopen(KERNEL_FILE, "r");
    if (!f) {
        std::fprintf(stderr, "could not open %s, run from the folder holding it\n", KERNEL_FILE);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    std::fseek(f, 0, SEEK_END);
    size_t size = std::ftell(f);
    std::rewind(f);
    std::string src(size, '\0');
    if (std::fread(&src[0], 1, size, f) != size) die("reading kernel source", 0);
    std::fclose(f);

    const char* text = src.c_str();
    cl_int err;
    cl_program program = clCreateProgramWithSource(ctx, 1, &text, &size, &err);
    if (err < 0) die("clCreateProgramWithSource", err);

    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (err < 0) {
        size_t logSize = 0;
        clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG, 0, NULL, &logSize);
        std::string log(logSize, '\0');
        clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG, logSize, &log[0], NULL);
        std::fprintf(stderr, "kernel build log:\n%s\n", log.c_str());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    return program;
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, procs, nameLen;
    char host[MPI_MAX_PROCESSOR_NAME];
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &procs);
    MPI_Get_processor_name(host, &nameLen);

    int N = (argc > 1) ? std::atoi(argv[1]) : 1000;
    if (N <= 0 || procs > N) {
        if (rank == 0)
            std::fprintf(stderr, "need N > 0 and at most one process per row (N=%d, procs=%d)\n",
                         N, procs);
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) matmul::announce("mpi+ocl", N, procs, 0);

    matmul::Bands b(N, procs, rank);
    std::vector<int> A, B, localA;
    std::vector<long long> C, localC(static_cast<size_t>(b.myRows) * N, 0);

    double t0 = 0, tGen = 0;

    if (rank == 0) {
        matmul::step("generating inputs");
        t0 = MPI_Wtime();
        matmul::generate(N, A, B);
        C.assign(static_cast<size_t>(N) * N, 0);
        tGen = MPI_Wtime();
    }

    if (rank == 0) matmul::step("broadcasting B and scattering A");
    MPI_Barrier(MPI_COMM_WORLD);
    double tA = MPI_Wtime();
    matmul::distribute(b, A, B, localA);
    MPI_Barrier(MPI_COMM_WORLD);
    double tB = MPI_Wtime();

    if (rank == 0) matmul::step("multiplying on the OpenCL device");

    // Device setup is outside the timed compute phase. Building the kernel from
    // source on every launch is a fixed cost that has nothing to do with N.
    bool isGpu = false;
    cl_device_id device = pickDevice(isGpu);
    char deviceName[256] = {0};
    cl_uint computeUnits = 0;
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof deviceName, deviceName, NULL);
    clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof computeUnits, &computeUnits, NULL);

    cl_int err;
    cl_context ctx = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    if (err < 0) die("clCreateContext", err);
    cl_command_queue queue = clCreateCommandQueue(ctx, device, 0, &err);
    if (err < 0) die("clCreateCommandQueue", err);
    cl_program program = buildProgram(ctx, device);
    cl_kernel kernel = clCreateKernel(program, "matmul_band", &err);
    if (err < 0) die("clCreateKernel", err);

    MPI_Barrier(MPI_COMM_WORLD);
    double tSetup = MPI_Wtime();

    const size_t bytesA = static_cast<size_t>(b.myRows) * N * sizeof(int);
    const size_t bytesB = static_cast<size_t>(N) * N * sizeof(int);
    const size_t bytesC = static_cast<size_t>(b.myRows) * N * sizeof(cl_long);

    cl_mem bufA = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytesA,
                                 localA.data(), &err);
    if (err < 0) die("clCreateBuffer A", err);
    cl_mem bufB = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytesB,
                                 B.data(), &err);
    if (err < 0) die("clCreateBuffer B", err);
    cl_mem bufC = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, bytesC, NULL, &err);
    if (err < 0) die("clCreateBuffer C", err);

    int rows = b.myRows;
    clSetKernelArg(kernel, 0, sizeof(int), &rows);
    clSetKernelArg(kernel, 1, sizeof(int), &N);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufA);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), &bufB);
    clSetKernelArg(kernel, 4, sizeof(cl_mem), &bufC);

    double tUpload = MPI_Wtime();

    size_t global[2] = {static_cast<size_t>(b.myRows), static_cast<size_t>(N)};
    cl_event event = NULL;
    err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global, NULL, 0, NULL, &event);
    if (err < 0) die("clEnqueueNDRangeKernel", err);
    clWaitForEvents(1, &event);

    double tKernel = MPI_Wtime();

    err = clEnqueueReadBuffer(queue, bufC, CL_TRUE, 0, bytesC, localC.data(), 0, NULL, NULL);
    if (err < 0) die("clEnqueueReadBuffer", err);

    MPI_Barrier(MPI_COMM_WORLD);
    double tC = MPI_Wtime();

    if (rank == 0) matmul::step("gathering C");
    matmul::collect(b, localC, C);
    MPI_Barrier(MPI_COMM_WORLD);
    double tD = MPI_Wtime();

    std::printf("rank %d on %s computed rows %d..%d (%d rows) on %s \"%s\" with %u compute units\n",
                rank, host, b.rowDispls[rank], b.rowDispls[rank] + b.myRows - 1, b.myRows,
                isGpu ? "GPU" : "CPU", deviceName, computeUnits);
    std::fflush(stdout);

    if (rank == 0) {
        auto us = [](double s) { return static_cast<long>(s * 1e6); };
        std::printf("device setup (untimed): %ld us  upload: %ld us  kernel: %ld us  read: %ld us\n",
                    us(tSetup - tB), us(tUpload - tSetup), us(tKernel - tUpload), us(tC - tKernel));
        matmul::report("mpi+ocl", N, procs, static_cast<int>(computeUnits),
                       tGen - t0, tB - tA, tC - tSetup, tD - tC,
                       matmul::checksum(C), matmul::verify(N, A, B, C));
    }

    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseMemObject(bufA);
    clReleaseMemObject(bufB);
    clReleaseMemObject(bufC);
    clReleaseCommandQueue(queue);
    clReleaseContext(ctx);

    MPI_Finalize();
    return 0;
}
