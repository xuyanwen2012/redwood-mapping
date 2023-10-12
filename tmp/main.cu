#include <algorithm>
#include <bitset>
#include <cub/cub.cuh>
#include <random>

#include "brt.cuh"
#include "cuda_utils.cuh"
#include "morton.cuh"
#include "oct.cuh"

#define DEFINE_SYNC_KERNEL_WRAPPER(kernel_name, function_name, num_threads)    \
  template <typename... Args>                                                  \
  void function_name(const size_t num_items, Args... args) {                   \
    const auto num_blocks = (num_items + num_threads - 1) / num_threads;       \
    kernel_name<<<num_blocks, num_threads>>>(num_items, args...);              \
    HANDLE_ERROR(cudaDeviceSynchronize());                                     \
  }

#define DEFINE_CUB_WRAPPER(kernel, wrapper_name)                               \
  template <typename... Args> void wrapper_name(Args... args) {                \
    void *d_temp_storage = nullptr;                                            \
    size_t temp_storage_bytes = 0;                                             \
    kernel(d_temp_storage, temp_storage_bytes, args...);                       \
    cudaMalloc(&d_temp_storage, temp_storage_bytes);                           \
    kernel(d_temp_storage, temp_storage_bytes, args...);                       \
    cudaFree(d_temp_storage);                                                  \
    HANDLE_ERROR(cudaDeviceSynchronize());                                     \
  }

// ---------------------
//        Kernels
// ---------------------

DEFINE_SYNC_KERNEL_WRAPPER(convertMortonOnly_v2, TransformMortonSync, 256)
DEFINE_SYNC_KERNEL_WRAPPER(BuildRadixTreeKernel, BuildRadixTreeSync, 256)
DEFINE_SYNC_KERNEL_WRAPPER(CalculateEdgeCountKernel, EdgeCountSync, 256)
DEFINE_SYNC_KERNEL_WRAPPER(MakeNodesKernel, MakeOctreeNodesSync, 256)
DEFINE_SYNC_KERNEL_WRAPPER(LinkNodesKernel, LinkNodesSync, 256)

DEFINE_CUB_WRAPPER(cub::DeviceRadixSort::SortKeys, CubRadixSort);
DEFINE_CUB_WRAPPER(cub::DeviceSelect::Unique, CubUnique);
DEFINE_CUB_WRAPPER(cub::DeviceScan::InclusiveSum, CubPrefixSum);

int main() {
  PrintCudaDeviceInfo();
  constexpr auto num_elements = 10'000'000;
  // constexpr auto num_elements = 1280 * 720;

  const auto u_input = AllocateManaged<float3>(num_elements);
  const auto u_mortons = AllocateManaged<Code_t>(num_elements);
  const auto u_mortons_alt = AllocateManaged<Code_t>(num_elements);
  const auto u_num_selected_out = AllocateManaged<int>(1);

  // init random inputs
  constexpr auto min_coord = 0.0f;
  constexpr auto max_coord = 1024.0f;
  constexpr auto range = max_coord - min_coord;
  constexpr auto morton_encoder = MortonEncoder(min_coord, range);
  constexpr auto morton_decoder = MortonDecoder(min_coord, range);
  thread_local std::mt19937 gen(114514); // NOLINT(cert-msc51-cpp)
  static std::uniform_real_distribution dis(min_coord, range); // <float>
  std::generate_n(u_input, num_elements,
                  [&] { return make_float3(dis(gen), dis(gen), dis(gen)); });

  GpuWarmUp();

  TransformMortonSync(num_elements, u_input, u_mortons, morton_encoder);
  CubRadixSort(u_mortons, u_mortons_alt, num_elements);
  CubUnique(u_mortons_alt, u_mortons, u_num_selected_out, num_elements);

  const auto num_unique = *u_num_selected_out;
  const auto num_brt_nodes = num_unique - 1;

  const auto u_inner_nodes = AllocateManaged<brt::InnerNodes>(num_brt_nodes);
  const auto u_edge_count = AllocateManaged<int>(num_brt_nodes);
  const auto u_oc_offset = AllocateManaged<int>(num_brt_nodes + 1);

  BuildRadixTreeSync(num_brt_nodes, u_mortons_alt, u_inner_nodes);
  EdgeCountSync(num_brt_nodes, u_edge_count, u_inner_nodes);
  u_edge_count[0] = 1; // Root node counts as 1
  CubPrefixSum(u_edge_count, u_oc_offset + 1, num_brt_nodes);
  u_oc_offset[0] = 0;

  const auto num_oc_nodes = u_oc_offset[num_brt_nodes];

  const auto u_oc_nodes =
      AllocateManaged<oct::OctNode>(num_oc_nodes); // could be less

  const auto root_level = u_inner_nodes[0].delta_node / 3;
  Code_t root_prefix = u_mortons[0] >> (kCodeLen - (root_level * 3));

  u_oc_nodes[0].cornor =
      morton_decoder(root_prefix << (kCodeLen - (root_level * 3)));
  u_oc_nodes[0].cell_size = range;
  MakeOctreeNodesSync(num_brt_nodes, u_oc_nodes, u_oc_offset, u_edge_count,
                      u_mortons, u_inner_nodes, morton_decoder, root_level);

  LinkNodesSync(num_brt_nodes, u_oc_nodes, u_oc_offset, u_edge_count, u_mortons,
                u_inner_nodes);

  // Print out some stats

  std::cout << "num_unique:\t" << num_unique << std::endl;
  std::cout << "num_oc_nodes:\t" << num_oc_nodes << std::endl;
  std::cout << "sorted (unique):" << std::endl;
  constexpr auto num_to_peek = 10;

  for (auto i = 0; i < num_to_peek; ++i) {
    std::cout << i << ":\t" << u_mortons[i] << "\t"
              << morton_decoder(u_mortons[i]) << std::endl;
  }

  std::cout << "brt_nodes:" << std::endl;
  for (auto i = 0; i < num_to_peek; ++i) {
    std::cout << i << ":\t" << u_inner_nodes[i].left << ", "
              << u_inner_nodes[i].right << "\t(" << u_inner_nodes[i].delta_node
              << ")" << std::endl;
  }

  std::cout << "edge_count (oc_offset):" << std::endl;
  for (auto i = 0; i < num_to_peek; ++i) {
    std::cout << i << ":\t" << u_edge_count[i] << "\t(" << u_oc_offset[i] << ')'
              << std::endl;
  }

  std::cout << "oc_nodes:" << std::endl;
  for (auto i = 0; i < num_to_peek; ++i) {
    std::cout << i << ":\t" << u_oc_nodes[i].cornor << ", "
              << u_oc_nodes[i].cell_size << ", "
              << std::bitset<8>(u_oc_nodes[i].child_node_mask) << ", "
              << std::bitset<8>(u_oc_nodes[i].child_leaf_mask) << std::endl;
  }

  CheckTree(root_prefix, root_level * 3, u_oc_nodes, 0, u_mortons);

  cudaFree(u_input);
  cudaFree(u_mortons);
  cudaFree(u_mortons_alt);
  cudaFree(u_num_selected_out);
  cudaFree(u_inner_nodes);
  cudaFree(u_edge_count);
  cudaFree(u_oc_offset);
  cudaFree(u_oc_nodes);
  return 0;
}
