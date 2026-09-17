// SIT315 M3.T2C - bitonic sort, one stage per kernel launch.
//
// Bitonic sort is a sorting network: a fixed sequence of compare-and-swap pairs
// that does not depend on the data. That is what makes it suitable for a device
// with thousands of work-items and no cheap way to synchronise between them.
// Quicksort cannot be expressed this way, because where it recurses next
// depends on where the partition landed.
//
// The network is built from two nested loops on the host. k is the size of the
// bitonic sequence being merged and doubles from 2 up to n. j is the compare
// distance inside that merge and halves from k/2 down to 1. Every (k, j) pair
// is one launch of this kernel, so a chunk of 2^m elements takes m(m+1)/2
// launches. The host cannot fuse them: every launch has to see the previous one
// finished, and a kernel-wide barrier across work-groups does not exist.
//
// Each work-item owns index i and pairs with i^j. Only the lower of the two
// does the swap, so exactly half the work-items are idle in every launch. That
// is inherent to the network, not a bug.
//
// Direction: ((i & k) == 0) means this element sits in an ascending run of the
// current merge, so the pair is ordered ascending, and descending otherwise.
// Alternating the direction at each level is what makes the next merge see a
// bitonic sequence.
//
// n must be a power of two. The host pads with INT_MAX, which sorts to the tail
// and is discarded on read-back.

__kernel void bitonic_step(__global int* data, const uint j, const uint k) {
    const uint i = get_global_id(0);
    const uint ixj = i ^ j;

    if (ixj > i) {
        const int a = data[i];
        const int b = data[ixj];
        const bool ascending = ((i & k) == 0);

        if ((a > b) == ascending) {
            data[i] = b;
            data[ixj] = a;
        }
    }
}
