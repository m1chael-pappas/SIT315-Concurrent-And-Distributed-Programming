// The kernel is the function that runs on the OpenCL device (the GPU, or the CPU
// if no GPU is found). The host program compiles this file at runtime with
// clBuildProgram, then launches one instance of this function per work-item
// through clEnqueueNDRangeKernel. With a global size of SZ, SZ work-items run
// this function in parallel across the device's processing elements, and each
// one squares a single element of v in place.
__kernel void square_magnitude(const int size,
                      __global int* v) {

    // Thread identifiers
    // get_global_id(0) returns this work-item's position in dimension 0 of the
    // global index space, so it doubles as the array index this work-item owns.
    const int globalIndex = get_global_id(0);

    //uncomment to see the index each PE works on
    //printf("Kernel process index :(%d)\n ", globalIndex);

    v[globalIndex] = v[globalIndex] * v[globalIndex];
}

// Activity 2 part 3: parallel vector addition. Same idea as square_magnitude,
// but reads from two input buffers and writes to a third. The host launches
// one work-item per element, so work-item i computes c[i] = a[i] + b[i].
__kernel void vector_add(const int size,
                         __global const int* a,
                         __global const int* b,
                         __global int* c) {
    const int i = get_global_id(0);
    if (i < size) {
        c[i] = a[i] + b[i];
    }
}
