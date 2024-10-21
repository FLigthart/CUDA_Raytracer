#pragma once

#include "../public/bvh/bvh.h"
#include <cuda_runtime_api.h>
#include <iostream>

#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ );

inline void check_cuda(cudaError_t result, char const* const function, const char* const file, int const line)
{
    if (result)
    {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
            file << ":" << line << " '" << function << "' \n";

        cudaDeviceReset();
        exit(99);
    }
}

inline void InitializeTree(int size, bvhNode* d_bvhTree)
{
    int treeSize = 2 * size;
    checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&d_bvhTree), treeSize * sizeof(bvhNode)));
}