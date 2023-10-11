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

struct CubTempStorage {
  CubTempStorage() : d_temp_storage(nullptr), saved_temp_storage_bytes(0) {}

  ~CubTempStorage() { cudaFree(d_temp_storage); }

  void ReallocIfNecessary(const size_t new_size) {
    if (saved_temp_storage_bytes < new_size) {
      HANDLE_ERROR(cudaFree(d_temp_storage));
      HANDLE_ERROR(cudaMalloc(&d_temp_storage, new_size));
      saved_temp_storage_bytes = new_size;
      std::cout << "saved_temp_storage_bytes:\t" << new_size << std::endl;
    }
  }

  void *d_temp_storage;
  size_t saved_temp_storage_bytes;
};

// ---------------------
//        Kernels
// ---------------------

// Use these to generate a wrapper function for a GPU kernel
DEFINE_SYNC_KERNEL_WRAPPER(convertMortonOnly_v2, TransformMortonSync, 256)
DEFINE_SYNC_KERNEL_WRAPPER(BuildRadixTreeKernel, BuildRadixTreeSync, 256)
DEFINE_SYNC_KERNEL_WRAPPER(CalculateEdgeCountKernel, CalculateEdgeCountSync,
                           256)

template <typename... Args>
void CubRadixSort(CubTempStorage &storage, Args... args) {
  size_t temp_storage_bytes = 0;
  cub::DeviceRadixSort::SortKeys(storage.d_temp_storage, temp_storage_bytes,
                                 args...);
  std::cout << "--- bytes :\t" << temp_storage_bytes << std::endl;
  storage.ReallocIfNecessary(temp_storage_bytes);
  cub::DeviceRadixSort::SortKeys(storage.d_temp_storage, temp_storage_bytes,
                                 args...);
  HANDLE_ERROR(cudaDeviceSynchronize());
}

template <typename... Args>
void CubUnique(CubTempStorage &storage, Args... args) {
  size_t temp_storage_bytes = 0;
  cub::DeviceSelect::Unique(storage.d_temp_storage, temp_storage_bytes,
                            args...);
  std::cout << "--- bytes :\t" << temp_storage_bytes << std::endl;
  storage.ReallocIfNecessary(temp_storage_bytes);
  cub::DeviceSelect::Unique(storage.d_temp_storage, temp_storage_bytes,
                            args...);
  HANDLE_ERROR(cudaDeviceSynchronize());
}

int main() {
  // constexpr auto num_elements = 10'000'000;
  constexpr auto num_elements = 1280 * 720;

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
  static std::uniform_real_distribution dis(min_coord, range); // <float>
  std::generate_n(u_input, num_elements,
                  [&] { return make_float3(dis(gen), dis(gen), dis(gen)); });

  GpuWarmUp();

  void *d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;
  size_t saved_temp_storage_bytes = 0;

  TransformMortonSync(num_elements, u_input, u_mortons, morton_functor);

  // CubTempStorage temp_storage{};
  // CubRadixSort(temp_storage, u_mortons, u_mortons_alt, num_elements);
  // CubUnique(temp_storage, u_mortons_alt, u_mortons, u_num_selected_out,
  //           num_elements);

  // Sort by morton codes
  {
    HANDLE_ERROR(cub::DeviceRadixSort::SortKeys(d_temp_storage,
                                                temp_storage_bytes, u_mortons,
                                                u_mortons_alt, num_elements));
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
    if (saved_temp_storage_bytes < temp_storage_bytes) {
      HANDLE_ERROR(cudaFree(d_temp_storage));
      HANDLE_ERROR(cudaMalloc(&d_temp_storage, temp_storage_bytes));
      saved_temp_storage_bytes = temp_storage_bytes;
    }
    cub::DeviceSelect::Unique(d_temp_storage, temp_storage_bytes, u_mortons_alt,
                              u_mortons, u_num_selected_out, num_elements);
    HANDLE_ERROR(cudaDeviceSynchronize());
  }

  const auto num_unique = *u_num_selected_out;
  const auto num_brt_nodes = num_unique - 1;

  BuildRadixTreeSync(num_brt_nodes, u_mortons_alt, u_inner_nodes);
  CalculateEdgeCountSync(num_brt_nodes, u_edge_count, u_inner_nodes);

  // Print out some stats
  std::cout << "num_unique:\t" << num_unique << std::endl;
  std::cout << "mortons:" << std::endl;
  for (auto i = 0; i < 10; ++i) {
    std::cout << u_mortons[i] << std::endl;
  }
  std::cout << "sorted:" << std::endl;
  for (auto i = 0; i < 10; ++i) {
    std::cout << u_mortons_alt[i] << std::endl;
  }
  for (auto i = 0; i < 10; ++i) {
    std::cout << i << ":\t" << u_inner_nodes[i].left << ", "
              << u_inner_nodes[i].right << "\t(" << u_inner_nodes[i].delta_node
              << ")" << std::endl;
  }
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
  // cudaFree(d_temp_storage);
  return 0;
}
