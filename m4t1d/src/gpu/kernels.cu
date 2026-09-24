/*
 * citysim device code.
 *
 * Each device function has the same name and does the same integer arithmetic
 * as its Rust counterpart, so the GPU reproduces the CPU bit for bit. Constants
 * are not written here: gpu::prelude generates them from the Rust sources and
 * puts them in front of this file before NVRTC compiles it.
 */

/** One Philox-4x32 round, the device copy of `philox::round`. */
__device__ __forceinline__ uint4 philox_round(uint4 ctr, uint2 key)
{
    const unsigned int lo0 = PHILOX_M0 * ctr.x;
    const unsigned int lo1 = PHILOX_M1 * ctr.z;
    return make_uint4(__umulhi(PHILOX_M1, ctr.z) ^ ctr.y ^ key.x, lo1,
                      __umulhi(PHILOX_M0, ctr.x) ^ ctr.w ^ key.y, lo0);
}

/** Philox-4x32-10, the device copy of `philox::philox4x32_10`. */
__device__ uint4 philox4x32_10(uint4 ctr, uint2 key)
{
#pragma unroll
    for (unsigned int r = 0; r < PHILOX_ROUNDS; ++r) {
        if (r > 0) {
            key.x += PHILOX_W0;
            key.y += PHILOX_W1;
        }
        ctr = philox_round(ctr, key);
    }
    return ctr;
}

/** Philox of `n` independent counter and key pairs, one thread each. Used by the self-test. */
extern "C" __global__ void philox_batch(const uint4* ctr, const uint2* key, uint4* out, unsigned int n)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = philox4x32_10(ctr[i], key[i]);
    }
}

/** A kernel that does nothing, for measuring what one launch costs. */
extern "C" __global__ void empty()
{
}
