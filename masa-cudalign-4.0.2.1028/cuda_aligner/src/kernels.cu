#include <cuda_runtime.h>

// Constants from Rust
#define THREADS_COUNT 128
#define ALPHA 4
#define DNA_MATCH 1
#define DNA_MISMATCH -3
#define DNA_GAP_OPEN 3
#define DNA_GAP_EXT 2

// Texture references for sequences
texture<unsigned char, 1, cudaReadModeElementType> tex_seq0;
texture<unsigned char, 1, cudaReadModeElementType> tex_seq1;

__device__ int my_max(int a, int b)
{
    return (a > b) ? a : b;
}

__device__ int my_max3(int a, int b, int c)
{
    return my_max(a, my_max(b, c));
}

__device__ void update_block_statistics(
    int4 *block_stats,
    int tid,
    int matches,
    int mismatches,
    int gaps)
{
    atomicAdd(&block_stats[tid].x, matches);
    atomicAdd(&block_stats[tid].y, mismatches);
    atomicAdd(&block_stats[tid].z, gaps);
}

__global__ void alignment_kernel(
    int2 *busH,
    int4 *busV_h,
    int4 *busV_e,
    int3 *busV_o,
    int4 *block_result,
    int4 *block_stats,
    const int i0,
    const int i1,
    const int step,
    const int2 cut_block,
    const int alignment_mode,
    unsigned char *traceback_matrix)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int thread_count = blockDim.x;

    __shared__ int2 shared_busH[THREADS_COUNT];

    // Calculate positions
    int i = i0 + tid * ALPHA + bid * thread_count * ALPHA;
    int j = step - i;

    // Initialize scores
    int4 h = make_int4(0, 0, 0, 0);
    int4 e = make_int4(0, 0, 0, 0);
    int4 f = make_int4(0, 0, 0, 0);

    // Load values from global memory
    if (i >= i0 && i < i1)
    {
        h = busV_h[tid];
        e = busV_e[tid];
    }

    int local_matches = 0;
    int local_mismatches = 0;
    int local_gaps = 0;

    // Initialize scores based on alignment mode
    if (alignment_mode == ALIGNMENT_MODE_GLOBAL)
    {
        // Initialize with gap penalties for global alignment
        h = make_int4(
            -DNA_GAP_FIRST * tid * ALPHA,
            -DNA_GAP_FIRST * (tid * ALPHA + 1),
            -DNA_GAP_FIRST * (tid * ALPHA + 2),
            -DNA_GAP_FIRST * (tid * ALPHA + 3));
    }
    else
    {
        // Initialize with zeros for local alignment
        h = make_int4(0, 0, 0, 0);
    }

// Main alignment loop
#pragma unroll
    for (int k = 0; k < ALPHA; k++)
    {
        // Get sequence values
        unsigned char c0 = tex1Dfetch(tex_seq0, i + k);
        unsigned char c1 = tex1Dfetch(tex_seq1, j);

        // Calculate match/mismatch score
        int match = (c0 == c1) ? DNA_MATCH : -DNA_MISMATCH;

        // Calculate new H value
        int new_h = my_max3(
            h.x + match, // Match/mismatch
            e.x,         // Gap in sequence 1
            f.x          // Gap in sequence 0
        );

        // Calculate new E value (gap in sequence 1)
        e.x = my_max(
            h.x - DNA_GAP_FIRST, // Open new gap
            e.x - DNA_GAP_EXT    // Extend existing gap
        );

        // Calculate new F value (gap in sequence 0)
        f.x = my_max(
            h.x - DNA_GAP_FIRST, // Open new gap
            f.x - DNA_GAP_EXT    // Extend existing gap
        );

        // Shift values
        h.x = h.y;
        h.y = h.z;
        h.z = h.w;
        h.w = new_h;

        e.x = e.y;
        e.y = e.z;
        e.z = e.w;

        // Inside the main loop, update statistics
        if (c0 == c1)
        {
            local_matches++;
        }
        else
        {
            local_mismatches++;
        }

        if (e.x > h.x || f.x > h.x)
        {
            local_gaps++;
        }
    }

    // Store results back to global memory
    if (i >= i0 && i < i1)
    {
        busV_h[tid] = h;
        busV_e[tid] = e;

        // Store best score for this block
        block_result[bid] = h;

        update_block_statistics(
            block_stats,
            tid,
            local_matches,
            local_mismatches,
            local_gaps);

        // Store traceback information
        int tb_idx = i * gridDim.x + blockIdx.x;
        if (from_diagonal)
        {
            traceback_matrix[tb_idx] = 0; // Match/Mismatch
        }
        else if (from_left)
        {
            traceback_matrix[tb_idx] = 1; // Deletion
        }
        else
        {
            traceback_matrix[tb_idx] = 2; // Insertion
        }
    }

    // Handle semi-global alignment
    if (alignment_mode == ALIGNMENT_MODE_SEMI_GLOBAL)
    {
        // Don't penalize end gaps
        if (i == i1 - 1 || j == seq1_len - 1)
        {
            h.w = max(h.w, 0);
        }
    }
}

extern "C"
{
    // Wrapper function to launch kernel
    void launch_alignment_kernel(
        dim3 grid,
        dim3 threads,
        int2 *busH,
        int4 *busV_h,
        int4 *busV_e,
        int3 *busV_o,
        int4 *block_result,
        int4 *block_stats,
        int i0,
        int i1,
        int step,
        int2 cut_block,
        int alignment_mode,
        unsigned char *traceback_matrix)
    {
        alignment_kernel<<<grid, threads>>>(
            busH,
            busV_h,
            busV_e,
            busV_o,
            block_result,
            block_stats,
            i0,
            i1,
            step,
            cut_block,
            alignment_mode,
            traceback_matrix);
    }

    // Functions for texture management
    cudaError_t bind_sequence_textures(
        const unsigned char *seq0,
        size_t seq0_len,
        const unsigned char *seq1,
        size_t seq1_len)
    {
        cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<unsigned char>();

        cudaBindTexture(NULL, tex_seq0, seq0, channel_desc, seq0_len);
        cudaBindTexture(NULL, tex_seq1, seq1, channel_desc, seq1_len);

        return cudaGetLastError();
    }

    void unbind_sequence_textures()
    {
        cudaUnbindTexture(tex_seq0);
        cudaUnbindTexture(tex_seq1);
    }
}
