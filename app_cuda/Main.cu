
#include <cuda_runtime.h>

#include <Eigen/Dense>
#include <cub/cub.cuh>
#include <iostream>

#include "Kernels.cuh"
#include "Morton.hpp"
// #include "RadixTree.cuh"
#include "Usm.hpp"

struct MortonFunctor {
  constexpr MortonFunctor(const float min_coord, const float range)
      : min_coord(min_coord), range(range) {}

  __host__ __device__ __forceinline__ Code_t
  operator()(const Eigen::Vector3f& point) const {
    return PointToCode(point.x(), point.y(), point.z(), min_coord, range);
  }

  float min_coord, range;
};

__global__ void ConvertMortonOnly(const Eigen::Vector3f* input, Code_t* output,
                                  const int size, const MortonFunctor functor) {
  const auto i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < size) output[i] = functor(input[i]);
}

__global__ void Empty() {}

void CudaWarmUp() {
  for (auto i = 0; i < 5; ++i) Empty<<<1, 1>>>();
}

// struct RadixTree {
//   brt_cuda::InnerNodes* u_brt_nodes;
// };

__device__ __host__ __forceinline__ int Sign(const int val) noexcept {
  return (val > 0) - (val < 0);
}

__device__ __host__ __forceinline__ int Divide2Ceil(const int x) noexcept {
  return (x + 1) >> 1;
}

namespace brt_cuda {

__device__ __host__ __forceinline__ int MakeLeaf(const int index) noexcept {
  return index ^ ((-1 ^ index) & 1UL << (sizeof(int) * 8 - 1));
}

__device__ __host__ __forceinline__ int MakeInternal(const int index) noexcept {
  return index;
}

struct InnerNodes {
  // The number of bits in the morton code, this node represents in [Karras]
  int delta_node;

  // pointers
  int left = -1;  // can be either inner or leaf
  int right = -1;
  int parent = -1;
};

__device__ __forceinline__ int Delta(const Code_t* morton_keys, const int i,
                                     const int j) {
  constexpr auto unused_bits = 1;
  const auto li = morton_keys[i];
  const auto lj = morton_keys[j];
  return __clzll(li ^ lj) - unused_bits;
}

__device__ __forceinline__ int DeltaSafe(const int key_num,
                                         const Code_t* morton_keys, const int i,
                                         const int j) noexcept {
  return (j < 0 || j >= key_num) ? -1 : Delta(morton_keys, i, j);
}

__device__ __forceinline__ void ProcessInternalNodesHelper(
    const int key_num, const Code_t* morton_keys, const int i,
    InnerNodes* brt_cuda_nodes) {
  // printf("i == 0\n");

  const auto direction{Sign(Delta(morton_keys, i, i + 1) -
                            DeltaSafe(key_num, morton_keys, i, i - 1))};

  const auto delta_min{DeltaSafe(key_num, morton_keys, i, i - direction)};

  int I_max{2};
  while (DeltaSafe(key_num, morton_keys, i, i + I_max * direction) > delta_min)
    I_max <<= 2;  // aka, *= 2

  // Find the other end using binary search.
  int I{0};
  for (int t{I_max / 2}; t; t /= 2)
    if (DeltaSafe(key_num, morton_keys, i, i + (I + t) * direction) > delta_min)
      I += t;

  const int j{i + I * direction};

  // Find the split position using binary search.
  const auto delta_node{DeltaSafe(key_num, morton_keys, i, j)};
  auto s{0};

  int t{I};
  do {
    t = Divide2Ceil(t);
    if (DeltaSafe(key_num, morton_keys, i, i + (s + t) * direction) >
        delta_node)
      s += t;
  } while (t > 1);

  const auto split{i + s * direction + min(direction, 0)};

  const int left{min(i, j) == split ? MakeLeaf(split) : MakeInternal(split)};
  const int right{max(i, j) == split + 1 ? MakeLeaf(split + 1)
                                         : MakeInternal(split + 1)};

  brt_cuda_nodes[i].delta_node = delta_node;
  brt_cuda_nodes[i].left = left;
  brt_cuda_nodes[i].right = right;

  if (min(i, j) != split) brt_cuda_nodes[left].parent = i;
  if (max(i, j) != split + 1) brt_cuda_nodes[right].parent = i;
}

}  // namespace brt_cuda

__global__ void BuildRadixTreeKernel(const Code_t* sorted_morton,
                                     brt_cuda::InnerNodes* nodes,
                                     const size_t num_unique_keys) {
  const auto i = threadIdx.x + blockIdx.x * blockDim.x;
  printf("i == %d\n", i);
  // const auto num_brt_nodes = num_unique_keys - 1;
  // if (i < num_brt_nodes) {
  //   brt_cuda::ProcessInternalNodesHelper(num_unique_keys, sorted_morton, i,
  //                                        nodes);
  // }
}

int main() {
  constexpr auto num_elements = 1280 * 720;
  constexpr auto min_coord = 0.0f;
  constexpr auto max_coord = 1024.0f;
  constexpr auto range = max_coord - min_coord;

  constexpr MortonFunctor morton_functor(min_coord, range);

  int deviceCount;
  HANDLE_ERROR(cudaGetDeviceCount(&deviceCount));

  for (int device = 0; device < deviceCount; ++device) {
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device);

    std::cout << "Device " << device << ": " << deviceProp.name << std::endl;
    std::cout << "Compute Capability (SM version): " << deviceProp.major << "."
              << deviceProp.minor << std::endl;
  }

  auto u_input = redwood::UsmMalloc<Eigen::Vector3f>(num_elements);
  auto u_morton_keys = redwood::UsmMalloc<Code_t>(num_elements);
  auto u_sorted_morton_keys = redwood::UsmMalloc<Code_t>(num_elements);
  auto d_num_selected_out = redwood::UsmMalloc<int>(1);

  auto u_brt_nodes = redwood::UsmMalloc<brt_cuda::InnerNodes>(num_elements);

  CudaWarmUp();

  // Prepare Inputs
  constexpr auto num_threads = 512;
  const auto num_blocks = (num_elements + num_threads - 1) / num_threads;
  GenerateRandomVector3f<<<num_blocks, num_threads>>>(u_input, 114514,
                                                      num_elements);
  HANDLE_ERROR(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);

  // Transform
  TimeTask("ConvertMortonOnly", [&] {
    ConvertMortonOnly<<<num_blocks, num_threads>>>(
        u_input, u_morton_keys, num_elements, morton_functor);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
  });

  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  std::cout << "Elapsed time: " << milliseconds << " ms" << std::endl;

  // Sort
  void* d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;
  size_t last_temp_storage_bytes = 0;

  HANDLE_ERROR(cub::DeviceRadixSort::SortKeys(
      d_temp_storage, temp_storage_bytes, u_morton_keys, u_sorted_morton_keys,
      num_elements));

  // Printout the temp storage
  std::cout << "temp_storage_bytes:\t" << temp_storage_bytes << std::endl;
  HANDLE_ERROR(cudaMalloc(&d_temp_storage, temp_storage_bytes));
  last_temp_storage_bytes = temp_storage_bytes;

  HANDLE_ERROR(cub::DeviceRadixSort::SortKeys(
      d_temp_storage, temp_storage_bytes, u_morton_keys, u_sorted_morton_keys,
      num_elements));

  HANDLE_ERROR(cudaDeviceSynchronize());

  // Unique
  cub::DeviceSelect::Unique(d_temp_storage, temp_storage_bytes,
                            u_sorted_morton_keys, u_morton_keys,
                            d_num_selected_out, num_elements);

  if (last_temp_storage_bytes < temp_storage_bytes) {
    HANDLE_ERROR(cudaFree(d_temp_storage));
    HANDLE_ERROR(cudaMalloc(&d_temp_storage, temp_storage_bytes));
    std::cout << "temp_storage_bytes:\t" << temp_storage_bytes << std::endl;
    last_temp_storage_bytes = temp_storage_bytes;
  }

  cub::DeviceSelect::Unique(d_temp_storage, temp_storage_bytes,
                            u_sorted_morton_keys, u_morton_keys,
                            d_num_selected_out, num_elements);

  HANDLE_ERROR(cudaDeviceSynchronize());

  const auto num_unique = *d_num_selected_out;

  std::cout << "num_unique:\t" << num_unique << std::endl;

  for (auto i = 0; i < 32; ++i) {
    std::cout << u_morton_keys[i] << std::endl;
  }

  // Now, Tree construction related stuff
  // const auto num_blocks = (num_unique + num_threads - 1) / num_threads;

  // Print out the parameters
  std::cout << "num_threads:\t" << num_threads << std::endl;
  std::cout << "num_blocks:\t" << num_blocks << std::endl;
  BuildRadixTreeKernel<<<num_threads, num_blocks>>>(u_morton_keys, u_brt_nodes,
                                                    num_unique);
  HANDLE_ERROR(cudaDeviceSynchronize());

  // Print out some brt nodes
  for (auto i = 0; i < 32; ++i) {
    std::cout << i << ":\t" << u_brt_nodes[i].left << ", "
              << u_brt_nodes[i].right << "\t(" << u_brt_nodes[i].delta_node
              << ")" << std::endl;
  }

  // Free
  redwood::UsmFree(u_input);
  redwood::UsmFree(u_morton_keys);
  redwood::UsmFree(u_sorted_morton_keys);
  redwood::UsmFree(d_num_selected_out);
  redwood::UsmFree(u_brt_nodes);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  return 0;
}
