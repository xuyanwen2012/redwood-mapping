#include <cstudio>

inline int _ConvertSMVer2Cores(int major, int minor) {
  // Defines for GPU Architecture types (using the SM version to determine
  // the # of cores per SM
  typedef struct {
    int SM; // 0xMm (hexidecimal notation), M = SM Major version,
    // and m = SM minor version
    int Cores;
  } sSMtoCores;

  sSMtoCores nGpuArchCoresPerSM[] = {
      {0x30, 192}, {0x32, 192}, {0x35, 192}, {0x37, 192}, {0x50, 128},
      {0x52, 128}, {0x53, 128}, {0x60, 64},  {0x61, 128}, {0x62, 128},
      {0x70, 64},  {0x72, 64},  {0x75, 64},  {0x80, 64},  {0x86, 128},
      {0x87, 128}, {0x89, 128}, {0x90, 128}, {-1, -1}};

  int index = 0;

  while (nGpuArchCoresPerSM[index].SM != -1) {
    if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor)) {
      return nGpuArchCoresPerSM[index].Cores;
    }

    index++;
  }
};

int main() {
  int deviceCount;
  cudaGetDeviceCount(&deviceCount);

  for (int device = 0; device < deviceCount; ++device) {
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device);

    printf("GPU Device %d: %s\n", device, deviceProp.name);
    printf("  Number of CUDA Cores: %d\n",
           deviceProp.multiProcessorCount *
               _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor));
    printf("  GPU Clock Rate (MHz): %d\n", deviceProp.clockRate / 1000);
    printf("  Memory Clock Rate (MHz): %d\n",
           deviceProp.memoryClockRate / 1000);
    printf("  L2 Cache Size: %d KiB\n", deviceProp.l2CacheSize / 1024);
    printf("  Total Global Memory: %lu MiB\n",
           deviceProp.totalGlobalMem / (1024 * 1024));
    printf("  Shared Memory per Block: %lu KiB\n",
           deviceProp.sharedMemPerBlock / 1024);
    printf("  Max Threads per Block: %d\n", deviceProp.maxThreadsPerBlock);
    printf("  Max Grid Dimensions: (%d, %d, %d)\n", deviceProp.maxGridSize[0],
           deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
    printf("  Max Threads per Dimension: (%d, %d, %d)\n",
           deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1],
           deviceProp.maxThreadsDim[2]);
    printf("\n");
  }

  return 0;
}
