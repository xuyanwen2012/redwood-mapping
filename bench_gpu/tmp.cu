#include <algorithm>
#include <cub/cub.cuh>
#include <iostream>
#include <random>

#include "brt.cuh"
#include "morton.cuh"
#include "oct.cuh"

#define HANDLE_ERROR(err) (HandleCudaError(err, __FILE__, __LINE__))

inline void HandleCudaError(const cudaError_t err, const char *file,
                            const int line) {
  if (err != cudaSuccess) {
    const auto _ = fprintf(stderr, "CUDA Error: %s in %s at line %d\n",
                           cudaGetErrorString(err), file, line);
    exit(EXIT_FAILURE);
  }
}
__global__ void EmptyKernel() {}

void GpuWarmUp() {
  EmptyKernel<<<1, 1>>>();
  HANDLE_ERROR(cudaDeviceSynchronize());
}

void PrintCudaDeviceInfo() {
  int device_count;
  HANDLE_ERROR(cudaGetDeviceCount(&device_count));

  for (int device = 0; device < device_count; ++device) {
    cudaDeviceProp device_prop;
    cudaGetDeviceProperties(&device_prop, device);

    std::cout << "Device " << device << ": " << device_prop.name << std::endl;
    std::cout << "Compute Capability (SM version): " << device_prop.major << "."
              << device_prop.minor << std::endl;
  }
}

template <typename T> T *AllocateManaged(const size_t num_elements) {
  T *ptr;
  HANDLE_ERROR(cudaMallocManaged(&ptr, num_elements * sizeof(T)));
  return ptr;
}

#define DEFINE_SYNC_KERNEL_WRAPPER(kernel_name, function_name, num_threads)    \
  template <typename... Args>                                                  \
  void function_name(const size_t num_items, Args... args) {                   \
    const auto num_blocks = (num_items + num_threads - 1) / num_threads;       \
    kernel_name<<<num_blocks, num_threads>>>(num_items, args...);              \
    HANDLE_ERROR(cudaDeviceSynchronize());                                     \
  }

// Use these to generate a wrapper function for a GPU kernel
DEFINE_SYNC_KERNEL_WRAPPER(convertMortonOnly_v2, TransformMortonSync, 256)
DEFINE_SYNC_KERNEL_WRAPPER(BuildRadixTreeKernel, BuildRadixTreeSync, 256)
DEFINE_SYNC_KERNEL_WRAPPER(CalculateEdgeCountKernel, CalculateEdgeCountSync,
                           256)

// ---------------------
//        Kernels
// ---------------------

int main() {
  constexpr auto num_elements = 10'000'000;
  //   constexpr auto num_elements = 1280 * 720;
  // constexpr auto num_threads = 256;

  PrintCudaDeviceInfo();

  const auto u_input = AllocateManaged<float3>(num_elements);
  const auto u_mortons = AllocateManaged<Code_t>(num_elements);
  const auto u_mortons_alt = AllocateManaged<Code_t>(num_elements);
  const auto u_num_selected_out = AllocateManaged<int>(1);
  const auto u_inner_nodes = AllocateManaged<brt::InnerNodes>(num_elements);
  const auto u_edge_count = AllocateManaged<int>(num_elements);
  const auto u_oc_offset = AllocateManaged<int>(num_elements);

  // init random inputs
  constexpr auto min_coord = 0.0f;
  constexpr auto max_coord = 1024.0f;
  constexpr auto range = max_coord - min_coord;
  constexpr auto morton_functor = Morton(min_coord, range);

  thread_local std::mt19937 gen(114514); // NOLINT(cert-msc51-cpp)
  static std::uniform_real_distribution<float> dis(min_coord, range);

  std::generate_n(u_input, num_elements,
                  [&] { return make_float3(dis(gen), dis(gen), dis(gen)); });

  GpuWarmUp();

  TransformMortonSync(num_elements, u_input, u_mortons, morton_functor);
  // FooWrapper(num_elements, u_input, u_mortons, morton_functor);

  // print 10
  std::cout << "mortons:" << std::endl;
  for (auto i = 0; i < 10; ++i) {
    std::cout << u_mortons[i] << std::endl;
  }

  void *d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;
  size_t last_temp_storage_bytes = 0;

  // Sort by morton codes
  {
    HANDLE_ERROR(cub::DeviceRadixSort::SortKeys(d_temp_storage,
                                                temp_storage_bytes, u_mortons,
                                                u_mortons_alt, num_elements));

    std::cout << "temp_storage_bytes:\t" << temp_storage_bytes << std::endl;
    HANDLE_ERROR(cudaMalloc(&d_temp_storage, temp_storage_bytes));

    HANDLE_ERROR(cub::DeviceRadixSort::SortKeys(d_temp_storage,
                                                temp_storage_bytes, u_mortons,
                                                u_mortons_alt, num_elements));

    HANDLE_ERROR(cudaDeviceSynchronize());
  }

  // Unique morton codes
  {
    cub::DeviceSelect::Unique(d_temp_storage, temp_storage_bytes, u_mortons_alt,
                              u_mortons, u_num_selected_out, num_elements);

    if (last_temp_storage_bytes < temp_storage_bytes) {
      HANDLE_ERROR(cudaFree(d_temp_storage));
      HANDLE_ERROR(cudaMalloc(&d_temp_storage, temp_storage_bytes));
      std::cout << "temp_storage_bytes:\t" << temp_storage_bytes << std::endl;
      last_temp_storage_bytes = temp_storage_bytes;
    }

    cub::DeviceSelect::Unique(d_temp_storage, temp_storage_bytes, u_mortons_alt,
                              u_mortons, u_num_selected_out, num_elements);

    HANDLE_ERROR(cudaDeviceSynchronize());
  }

  const auto num_unique = *u_num_selected_out;
  std::cout << "num_unique:\t" << num_unique << std::endl;

  // print 10
  std::cout << "sorted:" << std::endl;
  for (auto i = 0; i < 10; ++i) {
    std::cout << u_mortons_alt[i] << std::endl;
  }

  // Build Radix Tree
  const auto num_brt_nodes = num_unique - 1;

  BuildRadixTreeSync(num_brt_nodes, u_mortons_alt, u_inner_nodes);

  // Print out some brt nodes
  for (auto i = 0; i < 10; ++i) {
    std::cout << i << ":\t" << u_inner_nodes[i].left << ", "
              << u_inner_nodes[i].right << "\t(" << u_inner_nodes[i].delta_node
              << ")" << std::endl;
  }

  CalculateEdgeCountSync(num_brt_nodes, u_edge_count, u_inner_nodes);

  // Print out some edge counts
  std::cout << "edge_count:" << std::endl;
  for (auto i = 0; i < 10; ++i) {
    std::cout << i << ":\t" << u_edge_count[i] << std::endl;
  }

  cudaFree(u_input);
  cudaFree(u_mortons);
  cudaFree(u_mortons_alt);
  cudaFree(u_num_selected_out);
  cudaFree(u_inner_nodes);
  cudaFree(u_edge_count);
  cudaFree(u_oc_offset);
  cudaFree(d_temp_storage);
  return 0;
}
