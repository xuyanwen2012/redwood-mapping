
#include <cuda_runtime.h>

#include <algorithm>
#include <cub/cub.cuh>
// #include <cub/device/device_merge_sort.cuh>
#include <iostream>
#include <numeric>

struct TripleDoubler {
  __host__ __device__ __forceinline__ int operator()(const int &a) const {
    return int(a * 3);
  }
};

int main() {
  int num_elements = 1024;  // Define the number of elements in your array
  int *d_input, *d_output;
  size_t num_bytes = num_elements * sizeof(int);

  cudaMallocManaged((void **)&d_input, num_bytes);
  cudaMallocManaged((void **)&d_output, num_bytes);

  // Initialize or copy your input data to d_input here

  std::iota(d_input, d_input + num_elements, 0);
  std::reverse(d_input, d_input + num_elements);

  TripleDoubler conversion_op;
  cub::TransformInputIterator<int, TripleDoubler, int *> itr(d_input,
                                                             conversion_op);

  void *d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;

  cub::DeviceMergeSort::SortKeysCopy(d_temp_storage, temp_storage_bytes, itr,
                                     d_output, num_elements);

  cudaMalloc(&d_temp_storage, temp_storage_bytes);

  cub::DeviceMergeSort::SortKeysCopy(d_temp_storage, temp_storage_bytes, itr,
                                     d_output, num_elements);

  // cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, itr,
  //                                d_output, num_elements);

  //   // Allocate temporary storage
  //   cudaMalloc(&d_temp_storage, temp_storage_bytes);

  //   // Call SortKeys twice
  //   cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, itr,
  //                                  d_output, num_elements);

  //   cudaDeviceSynchronize();

  for (int i = 0; i < num_elements; i++) {
    std::cout << i << ":\t" << d_output[i] << "\n";
  }

  cudaFree(d_input);
  cudaFree(d_output);
  //   cudaFree(d_temp_storage);

  return 0;
}
