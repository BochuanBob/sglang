// Feasible graph-native solvers for online replica dispatch.
// One warp solves one MoE layer instance. Topology and coloring are static.

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>
#include <sgl_kernel/utils.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace sglang {

constexpr int GRAPH_SOLVER_SWEEPS = 5;

template <int NUM_GPUS, int NUM_LOGICAL, int NUM_REP, int COPIES>
struct GraphSolverState {
  float load[NUM_GPUS];
  float flow[(NUM_REP > 0 ? NUM_REP : 1) * COPIES];
  int sweeps;
};

__device__ __forceinline__ float warp_max(float value) {
#pragma unroll
  for (int offset = 16; offset; offset >>= 1)
    value = fmaxf(value, __shfl_down_sync(0xffffffff, value, offset));
  return __shfl_sync(0xffffffff, value, 0);
}

template <int NUM_GPUS, int NUM_LOGICAL, int NUM_REP, int COPIES,
          bool WATERFILL>
__global__ void graph_lplb_kernel(
    float* __restrict__ probability, const float* __restrict__ counts,
    const int32_t* __restrict__ logical_rank,
    const int32_t* __restrict__ replicated_logical,
    const int32_t* __restrict__ eligible_ranks,
    const int32_t* __restrict__ valid_copies,
    const int32_t* __restrict__ colored_experts,
    const int32_t* __restrict__ color_offsets, int num_colors,
    int32_t* __restrict__ out_sweeps, float* __restrict__ out_max_load) {
  static_assert(NUM_GPUS <= 32, "graph LPLB uses one warp");
  static_assert(WATERFILL || COPIES == 2, "edge balance requires pairs");
  __shared__ GraphSolverState<NUM_GPUS, NUM_LOGICAL, NUM_REP, COPIES> state;
  const int lane = threadIdx.x;

  for (int rank = lane; rank < NUM_GPUS; rank += 32) state.load[rank] = 0.f;
  for (int i = lane; i < NUM_LOGICAL * COPIES; i += 32) probability[i] = 0.f;
  __syncwarp();

  // Single-copy experts form the fixed base load. A rank of -1 marks a
  // replicated expert, whose complete demand is initialized below.
  for (int logical = lane; logical < NUM_LOGICAL; logical += 32) {
    int rank = logical_rank[logical];
    if (rank >= 0) {
      atomicAdd(&state.load[rank], counts[logical]);
      probability[logical * COPIES] = 1.f;
    }
  }
  __syncwarp();

  for (int expert = lane; expert < NUM_REP; expert += 32) {
    float demand = counts[replicated_logical[expert]];
    int copies = valid_copies[expert];
#pragma unroll
    for (int slot = 0; slot < COPIES; ++slot) {
      if (slot >= copies) continue;
      float value = demand / copies;
      state.flow[expert * COPIES + slot] = value;
      atomicAdd(&state.load[eligible_ranks[expert * COPIES + slot]], value);
    }
  }
  if (lane == 0) state.sweeps = 0;
  __syncwarp();

  for (int sweep = 0; sweep < GRAPH_SOLVER_SWEEPS; ++sweep) {
    for (int color = 0; color < num_colors; ++color) {
      for (int position = color_offsets[color] + lane;
           position < color_offsets[color + 1]; position += 32) {
        int expert = colored_experts[position];
        int first = expert * COPIES;
        if constexpr (!WATERFILL) {
          int r0 = eligible_ranks[first];
          int r1 = eligible_ranks[first + 1];
          float f0 = state.flow[first];
          float f1 = state.flow[first + 1];
          float delta = 0.5f * (state.load[r1] - state.load[r0]);
          delta = fminf(fmaxf(delta, -f0), f1);
          state.flow[first] = f0 + delta;
          state.flow[first + 1] = f1 - delta;
          state.load[r0] += delta;
          state.load[r1] -= delta;
        } else {
          float old_flow[COPIES];
          float sorted_load[COPIES];
          int order[COPIES];
          int copies = valid_copies[expert];
          float demand = counts[replicated_logical[expert]];
#pragma unroll
          for (int slot = 0; slot < COPIES; ++slot) {
            if (slot >= copies) continue;
            int rank = eligible_ranks[first + slot];
            old_flow[slot] = state.flow[first + slot];
            state.load[rank] -= old_flow[slot];
            sorted_load[slot] = state.load[rank];
            order[slot] = slot;
          }
          // COPIES is normally 2-4, making an in-register insertion sort
          // cheaper than a general sorting primitive or another kernel.
#pragma unroll
          for (int i = 1; i < COPIES; ++i) {
            if (i >= copies) break;
            float key = sorted_load[i];
            int key_order = order[i];
            int j = i - 1;
            while (j >= 0 && sorted_load[j] > key) {
              sorted_load[j + 1] = sorted_load[j];
              order[j + 1] = order[j];
              --j;
            }
            sorted_load[j + 1] = key;
            order[j + 1] = key_order;
          }
          float remaining = demand;
          float level = sorted_load[0];
#pragma unroll
          for (int width = 1; width < COPIES; ++width) {
            if (width >= copies) break;
            float needed = (sorted_load[width] - level) * width;
            if (remaining <= needed) {
              level += remaining / width;
              remaining = 0.f;
              break;
            }
            remaining -= needed;
            level = sorted_load[width];
          }
          if (remaining > 0.f) level += remaining / copies;
          float new_flow[COPIES];
#pragma unroll
          for (int slot = 0; slot < COPIES; ++slot) new_flow[slot] = 0.f;
#pragma unroll
          for (int sorted = 0; sorted < COPIES; ++sorted)
            if (sorted < copies)
              new_flow[order[sorted]] =
                  fmaxf(level - sorted_load[sorted], 0.f);
#pragma unroll
          for (int slot = 0; slot < COPIES; ++slot) {
            if (slot >= copies) continue;
            int rank = eligible_ranks[first + slot];
            state.flow[first + slot] = new_flow[slot];
            state.load[rank] += new_flow[slot];
          }
        }
      }
      __syncwarp();
    }
    if (lane == 0) state.sweeps = sweep + 1;
    __syncwarp();
  }

  for (int expert = lane; expert < NUM_REP; expert += 32) {
    int logical = replicated_logical[expert];
#pragma unroll
    for (int slot = 0; slot < COPIES; ++slot)
      if (slot < valid_copies[expert])
        probability[logical * COPIES + slot] =
            state.flow[expert * COPIES + slot];
  }
  float local_max = -3.402823466e+38F;
  for (int rank = lane; rank < NUM_GPUS; rank += 32)
    local_max = fmaxf(local_max, state.load[rank]);
  float maximum = warp_max(local_max);
  if (lane == 0) {
    *out_sweeps = state.sweeps;
    *out_max_load = maximum;
  }
}

template <int NUM_GPUS, int NUM_LOGICAL, int NUM_REP, int COPIES,
          bool WATERFILL>
void graph_lplb(
    tvm::ffi::TensorView probability, tvm::ffi::TensorView counts,
    tvm::ffi::TensorView logical_rank,
    tvm::ffi::TensorView replicated_logical,
    tvm::ffi::TensorView eligible_ranks,
    tvm::ffi::TensorView valid_copies,
    tvm::ffi::TensorView colored_experts,
    tvm::ffi::TensorView color_offsets, int64_t num_colors,
    tvm::ffi::TensorView out_sweeps, tvm::ffi::TensorView out_max_load) {
  using namespace host;
  SymbolicDevice device;
  TensorMatcher({NUM_LOGICAL, COPIES}).with_dtype<float>().with_device<kDLCUDA>(device).verify(probability);
  TensorMatcher({NUM_LOGICAL}).with_dtype<float>().with_device<kDLCUDA>(device).verify(counts);
  TensorMatcher({NUM_LOGICAL}).with_dtype<int32_t>().with_device<kDLCUDA>(device).verify(logical_rank);
  TensorMatcher({NUM_REP}).with_dtype<int32_t>().with_device<kDLCUDA>(device).verify(replicated_logical).verify(colored_experts);
  TensorMatcher({NUM_REP}).with_dtype<int32_t>().with_device<kDLCUDA>(device).verify(valid_copies);
  TensorMatcher({NUM_REP, COPIES}).with_dtype<int32_t>().with_device<kDLCUDA>(device).verify(eligible_ranks);
  SymbolicSize NUM_OFFSETS{"num_color_offsets"};
  TensorMatcher({NUM_OFFSETS}).with_dtype<int32_t>().with_device<kDLCUDA>(device).verify(color_offsets);
  TensorMatcher({1}).with_dtype<int32_t>().with_device<kDLCUDA>(device).verify(out_sweeps);
  TensorMatcher({1}).with_dtype<float>().with_device<kDLCUDA>(device).verify(out_max_load);
  LaunchKernel(1, 32, device.unwrap())(
      graph_lplb_kernel<NUM_GPUS, NUM_LOGICAL, NUM_REP, COPIES, WATERFILL>,
      static_cast<float*>(probability.data_ptr()),
      static_cast<const float*>(counts.data_ptr()),
      static_cast<const int32_t*>(logical_rank.data_ptr()),
      static_cast<const int32_t*>(replicated_logical.data_ptr()),
      static_cast<const int32_t*>(eligible_ranks.data_ptr()),
      static_cast<const int32_t*>(valid_copies.data_ptr()),
      static_cast<const int32_t*>(colored_experts.data_ptr()),
      static_cast<const int32_t*>(color_offsets.data_ptr()),
      static_cast<int>(num_colors), static_cast<int32_t*>(out_sweeps.data_ptr()),
      static_cast<float*>(out_max_load.data_ptr()));
}

}  // namespace sglang
