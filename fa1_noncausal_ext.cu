// fa1_noncausal_ext.cu
// PyTorch extension wrapper around the non-causal FA1 kernel defined in
// fa1-non-causal-alternate-implemetation.cu, so it can be benchmarked from
// Python against F.scaled_dot_product_attention(..., is_causal=False).
//
// The kernel itself is untouched -- this file only adds the host-side launch
// config (grid/block/shared-mem sizing) and the pybind11 binding.

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>
#include <limits>

#include "fa1-non-causal-alternate-implemetation.cu"

#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))
#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_F32(x)  TORCH_CHECK((x).scalar_type() == at::kFloat, #x " must be float32")

// Q, K, V : (B, nh, N, d) contiguous fp32 CUDA -> O : (B, nh, N, d)
torch::Tensor fa1_noncausal_forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
    CHECK_CUDA(Q); CHECK_CUDA(K); CHECK_CUDA(V);
    CHECK_F32(Q);  CHECK_F32(K);  CHECK_F32(V);
    Q = Q.contiguous(); K = K.contiguous(); V = V.contiguous();

    constexpr int Br = 32, Bc = 32;   // Br*Bc = 1024 threads/block
    int B  = Q.size(0);
    int nh = Q.size(1);
    int N  = Q.size(2);
    int d  = Q.size(3);
    int Tr = CEIL_DIV(N, Br);
    int Tc = CEIL_DIV(N, Bc);

    // O accumulates in-place across key tiles, so it must start at zero.
    // l (running softmax denom) starts at 0, m (running max) starts at -inf,
    // matching what the kernel expects to read as the "old" state at j=0.
    auto O = torch::zeros_like(Q);
    auto l = torch::zeros({B, nh, N}, Q.options());
    auto m = torch::full({B, nh, N}, -std::numeric_limits<float>::infinity(), Q.options());

    float scale = 1.0f / std::sqrt((float)d);
    size_t smem = (size_t)(2 * Br * d + 2 * Bc * d + Br * Bc + 5 * Br) * sizeof(float);

    // Default static shared-mem limit is 48KB on most archs; opt in to more
    // if a larger head_dim ever pushes past that.
    if (smem > 48 * 1024) {
        cudaFuncSetAttribute(flash_attn_1_kernel<Br, Bc>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    }

    dim3 grid(B, nh);
    dim3 block(Br * Bc);
    flash_attn_1_kernel<Br, Bc><<<grid, block, smem>>>(
        Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(),
        N, d, Tr, Tc, scale,
        l.data_ptr<float>(), m.data_ptr<float>(), O.data_ptr<float>());

    return O;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, mod) {
    mod.def("fa1_noncausal_forward", &fa1_noncausal_forward,
            "Non-causal Flash Attention v1 forward (fp32)");
}
