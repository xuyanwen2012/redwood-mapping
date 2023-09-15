#pragma once

// Platform-specific includes
#ifdef _WIN32
#include <intrin.h>
#elif defined(__CUDACC__)
#include <cuda_runtime.h>
#else
#error Unsupported compiler or platform
#endif

// Count leading zeros
#ifdef _WIN32
#define CLZ64(x) __lzcnt64(x)
#elif defined(__GNUC__) || defined(__clang__)
#define CLZ64(x) __builtin_clzll(x)
#elif defined(__CUDACC__)
#define CLZ64(x) __clzll(x)
#else
#error Unsupported compiler or platform
#endif
