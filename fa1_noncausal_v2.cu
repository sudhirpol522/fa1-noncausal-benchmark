// fa1_noncausal_v2.cu
// Optimized non-causal flash-attention forward (still fp32, still one output
// element of S per thread), fixing the three main bottlenecks of the v1 kernel:
//
//   1. FA2-style loop order: one block per (batch, head, Q-tile). O, l, m live
//      in shared memory / registers for the whole kernel and O is written to
//      global memory exactly once, instead of being read+rescaled+rewritten
//      for every key tile. This also multiplies grid parallelism by Tr.
//   2. K tile is padded to (d+1) floats per row in shared memory, so the
//      QK^T inner loop is bank-conflict-free (v1 had a 32-way conflict).
//   3. Softmax statistics (row max / row sum) use warp shuffle reductions with
//      all 32 lanes participating, instead of serializing on lane 0.
//
// Thread mapping: block = Br*Bc threads, warp w owns query row w of the tile
// (requires Bc == 32), lane c owns key column c of the current key tile.

template <const int Br, const int Bc>
__global__ void flash_attn_v2_kernel(const float* Q, const float* K, const float* V,
                                     int N, int d, int Tc, float scale, float* O) {
    static_assert(Bc == 32, "warp-per-row mapping requires Bc == warp size");

    int tid   = threadIdx.x;
    int s_row = tid / Bc;                 // warp index == query row in this tile
    int s_col = tid % Bc;                 // lane index == key column in the tile
    int qtile = blockIdx.x;
    int head  = blockIdx.y;
    int sample = blockIdx.z;

    long qkv_off = ((long)sample * gridDim.y + head) * (long)N * d;
    int global_row = qtile * Br + s_row;

    extern __shared__ float smem[];
    float* Qi  = smem;                    // Br x d
    float* Kj  = Qi  + Br * d;            // Bc x (d+1), padded row stride
    float* Vj  = Kj  + Bc * (d + 1);      // Bc x d
    float* Sij = Vj  + Bc * d;            // Br x Bc
    float* Oi  = Sij + Br * Bc;           // Br x d, accumulated across key tiles

    // Q tile is loaded once per block (v1 reloaded it for every key tile).
    for (int idx = tid; idx < Br * d; idx += Br * Bc) {
        int row = idx / d, col = idx % d;
        Qi[idx] = (qtile * Br + row < N)
                      ? Q[qkv_off + (long)(qtile * Br + row) * d + col] : 0.f;
        Oi[idx] = 0.f;
    }

    // Running softmax stats for this warp's row; uniform across the warp.
    float m_i = -INFINITY;
    float l_i = 0.f;

    for (int j = 0; j < Tc; j++) {
        // Also orders the initial Qi/Oi stores before any cross-warp reads.
        __syncthreads();
        for (int idx = tid; idx < Bc * d; idx += Br * Bc) {
            int row = idx / d, col = idx % d;
            bool valid = (j * Bc + row < N);
            Kj[row * (d + 1) + col] = valid ? K[qkv_off + (long)(j * Bc + row) * d + col] : 0.f;
            Vj[row * d + col]       = valid ? V[qkv_off + (long)(j * Bc + row) * d + col] : 0.f;
        }
        __syncthreads();

        // s = scale * <Q row, K col>; padded key columns masked to -inf.
        float acc = 0.f;
        for (int k = 0; k < d; k++)
            acc += Qi[s_row * d + k] * Kj[s_col * (d + 1) + k];
        float s = (j * Bc + s_col < N) ? acc * scale : -INFINITY;

        float row_m = s;
        for (int off = 16; off > 0; off >>= 1)
            row_m = fmaxf(row_m, __shfl_xor_sync(0xffffffffu, row_m, off));

        float m_new = fmaxf(m_i, row_m);
        float p = __expf(s - m_new);      // exactly 0 for masked columns
        float row_l = p;
        for (int off = 16; off > 0; off >>= 1)
            row_l += __shfl_xor_sync(0xffffffffu, row_l, off);

        float alpha = __expf(m_i - m_new);
        l_i = l_i * alpha + row_l;
        m_i = m_new;

        // Row s_row of Sij is written and read only by this warp.
        Sij[s_row * Bc + s_col] = p;
        __syncwarp();

        for (int col = s_col; col < d; col += Bc) {
            float pv = 0.f;
            for (int c = 0; c < Bc; c++)
                pv += Sij[s_row * Bc + c] * Vj[c * d + col];
            Oi[s_row * d + col] = Oi[s_row * d + col] * alpha + pv;
        }
    }

    if (global_row < N) {
        float inv_l = 1.f / l_i;
        for (int col = s_col; col < d; col += Bc)
            O[qkv_off + (long)global_row * d + col] = Oi[s_row * d + col] * inv_l;
    }
}
