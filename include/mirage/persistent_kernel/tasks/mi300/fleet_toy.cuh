#pragma once

#include <hip/hip_bfloat16.h>

namespace kernel {

using fleet_toy_bf16 = __hip_bfloat16;

template <int K>
__device__ __forceinline__ void fleet_toy_rms_row(void const *x_ptr,
                                                  void const *gamma_ptr,
                                                  void *h_ptr,
                                                  float epsilon) {
  auto const *x = static_cast<fleet_toy_bf16 const *>(x_ptr);
  auto const *gamma = static_cast<fleet_toy_bf16 const *>(gamma_ptr);
  auto *h = static_cast<fleet_toy_bf16 *>(h_ptr);
  __shared__ float partial[256];

  float sum_sq = 0.0f;
  for (int k = threadIdx.x; k < K; k += blockDim.x) {
    float value = static_cast<float>(x[k]);
    sum_sq += value * value;
  }
  partial[threadIdx.x] = sum_sq;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      partial[threadIdx.x] += partial[threadIdx.x + stride];
    }
    __syncthreads();
  }

  float inv_rms = rsqrtf(partial[0] / static_cast<float>(K) + epsilon);
  for (int k = threadIdx.x; k < K; k += blockDim.x) {
    h[k] = static_cast<fleet_toy_bf16>(
        static_cast<float>(x[k]) * inv_rms * static_cast<float>(gamma[k]));
  }
}

template <int M, int K, int N_PER_XCD>
__device__ __forceinline__ void
    fleet_toy_gang_linear(void const *h_ptr,
                          void const *weight_xcd_ptr,
                          void *y_xcd_ptr,
                          int output_stride,
                          int tile_idx) {
  auto const *h = static_cast<fleet_toy_bf16 const *>(h_ptr);
  auto const *weight =
      static_cast<fleet_toy_bf16 const *>(weight_xcd_ptr);
  auto *y = static_cast<fleet_toy_bf16 *>(y_xcd_ptr);

  if (tile_idx >= M) {
    return;
  }
  for (int n = threadIdx.x; n < N_PER_XCD; n += blockDim.x) {
    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
      acc += static_cast<float>(h[tile_idx * K + k]) *
             static_cast<float>(weight[n * K + k]);
    }
    y[tile_idx * output_stride + n] =
        static_cast<fleet_toy_bf16>(acc);
  }
}

template <int N>
__device__ __forceinline__ void fleet_toy_add_row(void const *y_ptr,
                                                  void const *residual_ptr,
                                                  void *out_ptr) {
  auto const *y = static_cast<fleet_toy_bf16 const *>(y_ptr);
  auto const *residual =
      static_cast<fleet_toy_bf16 const *>(residual_ptr);
  auto *out = static_cast<fleet_toy_bf16 *>(out_ptr);
  for (int n = threadIdx.x; n < N; n += blockDim.x) {
    out[n] = static_cast<fleet_toy_bf16>(
        static_cast<float>(y[n]) + static_cast<float>(residual[n]));
  }
}

} // namespace kernel
