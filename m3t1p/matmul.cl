// SIT315 M3.T1P - matrix multiplication kernel.
//
// One work-item computes one element of C. The index space is two dimensional:
// dimension 0 is the row within this rank's band, dimension 1 is the column.
// A rank that owns R rows launches R x N work-items, so the kernel replaces the
// two outer loops of the CPU version and keeps only the k loop.
//
// A and B hold int, C holds long, matching the host so a 1000 x 1000 product of
// values under 100 cannot overflow.

__kernel void matmul_band(const int rows,
                          const int N,
                          __global const int* A,
                          __global const int* B,
                          __global long* C) {
    const int i = get_global_id(0);
    const int j = get_global_id(1);

    // The runtime may round the global size up to a multiple of the work-group
    // size, so both dimensions need a bounds check.
    if (i >= rows || j >= N) return;

    long sum = 0;
    for (int k = 0; k < N; ++k) {
        sum += (long)A[i * N + k] * (long)B[k * N + j];
    }
    C[i * N + j] = sum;
}
