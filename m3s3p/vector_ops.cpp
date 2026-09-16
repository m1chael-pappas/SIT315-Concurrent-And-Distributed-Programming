#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include <CL/cl.h>

#define PRINT 1

int SZ = 8;
int *v;

// Activity 2 part 3: host vectors for the parallel vector addition (v3 = v1 + v2).
int *v1, *v2, *v3;

// Device-side buffer that mirrors the host vector v. The host copies v into it
// before the kernel runs and reads the squared values back out of it afterwards.
cl_mem bufV;
// Device-side copies of v1 and v2 (the kernel reads) and v3 (the kernel writes).
cl_mem bufV1, bufV2, bufV3;

// Handle to the device the kernel will run on (GPU first, CPU as the fallback).
cl_device_id device_id;
// The OpenCL context. It groups the device with the memory objects, programs,
// kernels and queues that belong to it. Every other OpenCL object is created
// inside this context.
cl_context context;
// The compiled program built from vector_ops.cl for the chosen device.
cl_program program;
// A handle to one __kernel function inside the program (square_magnitude here).
// This is what gets its arguments set and gets enqueued for execution.
cl_kernel kernel;
// The command queue for the device. The host pushes work onto it (buffer
// writes, kernel launches, buffer reads) and the device pulls commands off it.
cl_command_queue queue;
cl_event event = NULL;

int err;

// Finds an OpenCL platform, then asks it for a GPU device. If no GPU exists it
// falls back to a CPU device. Returns the device handle.
cl_device_id create_device();
// One-stop setup: picks the device, creates the context, builds the .cl file
// into a program, creates the command queue and pulls the named kernel out of
// the program.
void setup_openCL_device_context_queue_kernel(char *filename, char *kernelname);
// Reads the kernel source file from disk, creates a program object from that
// source and compiles it for the given device. Prints the compiler log and
// exits if the build fails.
cl_program build_program(cl_context ctx, cl_device_id dev, const char *filename);
// Allocates the device buffer bufV and copies the host vector v into it.
void setup_kernel_memory();
// Binds the kernel's arguments (the size and the device buffer) to the kernel
// object so the device has them when the kernel runs.
void copy_kernel_args();
// Releases the device buffer, the OpenCL objects and the host vector.
void free_memory();

void init(int *&A, int size);
void print(int *A, int size);

// Activity 2 part 3: runs the vector_add kernel over v1 and v2 and stores the
// result in v3, timing the copies and the kernel so it can be compared with the
// std::thread version in vector_add_threads.cpp.
void run_vector_add();

int main(int argc, char **argv)
{
    if (argc > 1)
        SZ = atoi(argv[1]);

    init(v, SZ);


    // The global work size: how many work-items to launch in total, per
    // dimension. We use a 1D index space of SZ work-items so each work-item
    // handles exactly one element of the vector.
    size_t global[1] = {(size_t)SZ};

    //initial vector
    print(v, SZ);

    setup_openCL_device_context_queue_kernel((char *)"./vector_ops.cl", (char *)"square_magnitude");

    setup_kernel_memory();
    copy_kernel_args();

    // clEnqueueNDRangeKernel puts a kernel launch on the command queue. The
    // arguments are:
    //   queue   - the command queue to submit to
    //   kernel  - the kernel object to run
    //   1       - number of dimensions in the index space (work_dim)
    //   NULL    - global work offset; NULL means the ids start at 0
    //   global  - global work size per dimension (SZ work-items in dimension 0)
    //   NULL    - local work size; NULL lets the runtime pick the work-group size
    //   0, NULL - number of events to wait for and the event list (none here)
    //   &event  - an event handle the runtime fills in, so we can wait on this
    //             launch with clWaitForEvents
    // The call is non-blocking, which is why clWaitForEvents follows it.
    clEnqueueNDRangeKernel(queue, kernel, 1, NULL, global, NULL, 0, NULL, &event);
    clWaitForEvents(1, &event);

    // clEnqueueReadBuffer copies data from a device buffer back into host
    // memory. The arguments are:
    //   queue            - the command queue to submit to
    //   bufV             - the device buffer to read from
    //   CL_TRUE          - blocking read; the call returns once the data is in v
    //   0                - byte offset into the buffer to start reading from
    //   SZ * sizeof(int) - number of bytes to read
    //   &v[0]            - host pointer to write the data into
    //   0, NULL, NULL    - no events to wait for, no event returned
    clEnqueueReadBuffer(queue, bufV, CL_TRUE, 0, SZ * sizeof(int), &v[0], 0, NULL, NULL);

    //result vector
    print(v, SZ);

    // Activity 2 part 3: parallel vector addition on the same context, queue
    // and program. The program already holds both kernels, so we only need a
    // new kernel handle and three more buffers.
    run_vector_add();

    //frees memory for device, kernel, queue, etc.
    //you will need to modify this to free your own buffers
    free_memory();
}

void run_vector_add()
{
    cl_kernel add_kernel;
    cl_event add_event = NULL;
    size_t global[1] = {(size_t)SZ};

    init(v1, SZ);
    init(v2, SZ);
    v3 = (int *)malloc(sizeof(int) * SZ);
    // Touch the output once so page faults on first write are not counted
    // as read-back time.
    memset(v3, 0, sizeof(int) * SZ);

    printf("vector addition\n");
    print(v1, SZ);
    print(v2, SZ);

    // Pull the second kernel out of the program that was already built.
    add_kernel = clCreateKernel(program, "vector_add", &err);
    if (err < 0)
    {
        perror("Couldn't create the vector_add kernel");
        exit(1);
    }

    // Timed region: allocate device buffers, copy inputs in, run the kernel,
    // copy the result out. This is the fair comparison with the threaded
    // version, since the threads work on host memory with no copies.
    auto t0 = std::chrono::high_resolution_clock::now();

    // The inputs are read-only from the kernel's point of view. Passing
    // CL_MEM_COPY_HOST_PTR copies v1/v2 into the buffers at creation, so no
    // separate clEnqueueWriteBuffer is needed. The output is write-only for the
    // kernel; the host reads it back afterwards.
    bufV1 = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, SZ * sizeof(int), v1, &err);
    bufV2 = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, SZ * sizeof(int), v2, &err);
    bufV3 = clCreateBuffer(context, CL_MEM_WRITE_ONLY, SZ * sizeof(int), NULL, &err);
    if (err < 0)
    {
        perror("Couldn't create a buffer");
        exit(1);
    }

    // Argument positions match vector_add(size, a, b, c) in vector_ops.cl.
    err = clSetKernelArg(add_kernel, 0, sizeof(int), (void *)&SZ);
    err |= clSetKernelArg(add_kernel, 1, sizeof(cl_mem), (void *)&bufV1);
    err |= clSetKernelArg(add_kernel, 2, sizeof(cl_mem), (void *)&bufV2);
    err |= clSetKernelArg(add_kernel, 3, sizeof(cl_mem), (void *)&bufV3);
    if (err < 0)
    {
        perror("Couldn't set a vector_add kernel argument");
        exit(1);
    }

    auto tk0 = std::chrono::high_resolution_clock::now();
    clEnqueueNDRangeKernel(queue, add_kernel, 1, NULL, global, NULL, 0, NULL, &add_event);
    clWaitForEvents(1, &add_event);
    auto tk1 = std::chrono::high_resolution_clock::now();

    clEnqueueReadBuffer(queue, bufV3, CL_TRUE, 0, SZ * sizeof(int), &v3[0], 0, NULL, NULL);
    auto t1 = std::chrono::high_resolution_clock::now();

    print(v3, SZ);

    // Check the device result against a plain host loop.
    for (long i = 0; i < SZ; i++)
    {
        if (v3[i] != v1[i] + v2[i])
        {
            printf("MISMATCH at %ld: %d + %d != %d\n", i, v1[i], v2[i], v3[i]);
            exit(1);
        }
    }
    printf("verified\n");

    auto us = [](auto a, auto b) {
        return std::chrono::duration_cast<std::chrono::microseconds>(b - a).count();
    };
    printf("size=%d kernel=%ld us copy+kernel+read=%ld us\n",
           SZ, us(tk0, tk1), us(t0, t1));

    clReleaseKernel(add_kernel);
}

void init(int *&A, int size)
{
    A = (int *)malloc(sizeof(int) * size);

    for (long i = 0; i < size; i++)
    {
        A[i] = rand() % 100; // any number less than 100
    }
}

void print(int *A, int size)
{
    if (PRINT == 0)
    {
        return;
    }

    if (PRINT == 1 && size > 15)
    {
        for (long i = 0; i < 5; i++)
        {                        //rows
            printf("%d ", A[i]); // print the cell value
        }
        printf(" ..... ");
        for (long i = size - 5; i < size; i++)
        {                        //rows
            printf("%d ", A[i]); // print the cell value
        }
    }
    else
    {
        for (long i = 0; i < size; i++)
        {                        //rows
            printf("%d ", A[i]); // print the cell value
        }
    }
    printf("\n----------------------------\n");
}

void free_memory()
{
    //free the buffers
    clReleaseMemObject(bufV);
    clReleaseMemObject(bufV1);
    clReleaseMemObject(bufV2);
    clReleaseMemObject(bufV3);

    //free opencl objects
    clReleaseKernel(kernel);
    clReleaseCommandQueue(queue);
    clReleaseProgram(program);
    clReleaseContext(context);

    free(v);
    free(v1);
    free(v2);
    free(v3);
}


void copy_kernel_args()
{
    // clSetKernelArg binds a value to one of the kernel's parameters, matched
    // by position. The arguments are:
    //   kernel       - the kernel object
    //   0 or 1       - the index of the parameter in the __kernel signature
    //   sizeof(...)  - size in bytes of the value being passed
    //   (void *)&... - pointer to the value. For a plain int this is the int
    //                  itself; for a buffer it is the cl_mem handle, which the
    //                  device sees as the __global int* pointer.
    // Argument 0 is the vector size and argument 1 is the device buffer, which
    // matches square_magnitude(const int size, __global int* v) in the .cl file.
    clSetKernelArg(kernel, 0, sizeof(int), (void *)&SZ);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&bufV);

    if (err < 0)
    {
        perror("Couldn't create a kernel argument");
        printf("error = %d", err);
        exit(1);
    }
}

void setup_kernel_memory()
{
    // clCreateBuffer allocates a memory object (a buffer) in device memory
    // inside the given context. The arguments are:
    //   context           - the context the buffer belongs to
    //   CL_MEM_READ_WRITE - memory flags (see below)
    //   SZ * sizeof(int)  - size of the buffer in bytes
    //   NULL              - host pointer; NULL because we copy the data in
    //                       separately with clEnqueueWriteBuffer
    //   NULL              - error code output, ignored here
    // The cl_mem_flags argument tells the runtime how the kernel will use the
    // buffer and where the initial data comes from. The main values are:
    //   CL_MEM_READ_WRITE     kernel can read and write it (the default)
    //   CL_MEM_READ_ONLY      kernel only reads it; writing from a kernel is
    //                         undefined behaviour
    //   CL_MEM_WRITE_ONLY     kernel only writes it
    //   CL_MEM_USE_HOST_PTR   the device uses the host pointer's memory directly
    //   CL_MEM_ALLOC_HOST_PTR allocate the buffer in host-accessible memory
    //   CL_MEM_COPY_HOST_PTR  allocate and copy the host pointer's data in now
    //   CL_MEM_HOST_WRITE_ONLY / CL_MEM_HOST_READ_ONLY / CL_MEM_HOST_NO_ACCESS
    //                         restrict what the host side may do with it
    // The read/write flags let the runtime place the buffer in the best memory
    // for that access pattern, and the host-pointer flags control whether data
    // is copied, shared or allocated fresh. We need READ_WRITE here because the
    // kernel reads v, squares it and writes it back into the same buffer.
    bufV = clCreateBuffer(context, CL_MEM_READ_WRITE, SZ * sizeof(int), NULL, NULL);

    // Copy matrices to the GPU
    clEnqueueWriteBuffer(queue, bufV, CL_TRUE, 0, SZ * sizeof(int), &v[0], 0, NULL, NULL);
}

void setup_openCL_device_context_queue_kernel(char *filename, char *kernelname)
{
    device_id = create_device();
    cl_int err;

    // clCreateContext creates the OpenCL context for the chosen device. The
    // context owns every buffer, program, kernel and queue we create after
    // this point, and the runtime uses it to manage memory and sync between
    // the host and the device. Arguments: no platform properties (NULL), one
    // device, the device list, no error callback (NULL, NULL) and an output
    // error code.
    context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &err);
    if (err < 0)
    {
        perror("Couldn't create a context");
        exit(1);
    }

    program = build_program(context, device_id, filename);

    // clCreateCommandQueueWithProperties creates the command queue that
    // connects the host to the device. Every write, read and kernel launch
    // goes through this queue and the device executes them in order (the
    // properties argument is 0, so out-of-order execution and profiling are
    // both off).
    queue = clCreateCommandQueueWithProperties(context, device_id, 0, &err);
    if (err < 0)
    {
        perror("Couldn't create a command queue");
        exit(1);
    };


    kernel = clCreateKernel(program, kernelname, &err);
    if (err < 0)
    {
        perror("Couldn't create a kernel");
        printf("error =%d", err);
        exit(1);
    };
}

cl_program build_program(cl_context ctx, cl_device_id dev, const char *filename)
{

    cl_program program;
    FILE *program_handle;
    char *program_buffer, *program_log;
    size_t program_size, log_size;

    /* Read program file and place content into buffer */
    program_handle = fopen(filename, "r");
    if (program_handle == NULL)
    {
        perror("Couldn't find the program file");
        exit(1);
    }
    fseek(program_handle, 0, SEEK_END);
    program_size = ftell(program_handle);
    rewind(program_handle);
    program_buffer = (char *)malloc(program_size + 1);
    program_buffer[program_size] = '\0';
    fread(program_buffer, sizeof(char), program_size, program_handle);
    fclose(program_handle);

    // clCreateProgramWithSource creates a program object from OpenCL C source
    // text held in host memory. The arguments are:
    //   ctx              - the context the program belongs to
    //   1                - number of source strings
    //   &program_buffer  - array of pointers to the source strings
    //   &program_size    - array of string lengths in bytes
    //   &err             - output error code
    // The program is not compiled yet at this point; clBuildProgram below does
    // that for the target device.
    program = clCreateProgramWithSource(ctx, 1,
                                        (const char **)&program_buffer, &program_size, &err);
    if (err < 0)
    {
        perror("Couldn't create the program");
        exit(1);
    }
    free(program_buffer);

    /* Build program 

   The fourth parameter accepts options that configure the compilation. 
   These are similar to the flags used by gcc. For example, you can 
   define a macro with the option -DMACRO=VALUE and turn off optimization 
   with -cl-opt-disable.
   */
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (err < 0)
    {

        /* Find size of log and print to std output */
        clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG,
                              0, NULL, &log_size);
        program_log = (char *)malloc(log_size + 1);
        program_log[log_size] = '\0';
        clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG,
                              log_size + 1, program_log, NULL);
        printf("%s\n", program_log);
        free(program_log);
        exit(1);
    }

    return program;
}

cl_device_id create_device() {

   cl_platform_id platform;
   cl_device_id dev;
   int err;

   /* Identify a platform */
   err = clGetPlatformIDs(1, &platform, NULL);
   if(err < 0) {
      perror("Couldn't identify a platform");
      exit(1);
   } 

   // Access a device
   // GPU
   err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &dev, NULL);
   if(err == CL_DEVICE_NOT_FOUND) {
      // CPU
      printf("GPU not found\n");
      err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &dev, NULL);
   }
   if(err < 0) {
      perror("Couldn't access any devices");
      exit(1);   
   }

   return dev;
}
