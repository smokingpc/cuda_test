
#include "cuda.h"
#include "nvml.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
//#include "nv-p2p.h"

#include <stdio.h>

int main()
{
    cudaError_t status = cudaSetDeviceFlags(cudaDeviceMapHost);
    status = cudaSetDevice(0);
    CUDA_POINTER_ATTRIBUTE_P2P_TOKENS tokens;
    CUdeviceptr ptr = 0;
    CUresult result = CUDA_SUCCESS;
    //cuDevicePrimaryCtxRetain();
    result = cuMemAlloc(&ptr, 4096);
    if(result == CUDA_SUCCESS)
    {
        result = cuPointerGetAttribute(&tokens, CU_POINTER_ATTRIBUTE_P2P_TOKENS, ptr);
        cuMemFree(ptr);
    }

    cudaDeviceSynchronize();
    cudaDeviceReset();
    return 0;
}
