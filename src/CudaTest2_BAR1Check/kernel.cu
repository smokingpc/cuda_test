
#include "cuda.h"
#include "nvml.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
//#include "nv-p2p.h"

#include <stdio.h>

int main()
{
    //cudaError_t status = cudaSetDeviceFlags(cudaDeviceMapHost);
    cudaError_t status = cudaSetDevice(0);
    CUdevice device;

    CUDA_POINTER_ATTRIBUTE_P2P_TOKENS tokens;
    CUdeviceptr ptr = 0;
    CUresult result = CUDA_SUCCESS;
    // Get handle for device 0
    result = cuDeviceGet(&device, 0);
    // Create context
    CUcontext context;
    cuCtxCreate(&context, 0, device);

    //cuDevicePrimaryCtxRetain();
    result = cuMemAlloc(&ptr, 4096);
    cudaDeviceSynchronize();
    if(result == CUDA_SUCCESS)
    {
        result = cuPointerGetAttribute(&tokens, CU_POINTER_ATTRIBUTE_P2P_TOKENS, ptr);
        if(result == CUDA_SUCCESS)
            printf("p2pToken=%lld, vaSpaceToken=%d\n", tokens.p2pToken, tokens.vaSpaceToken);
        else
            printf("cuPointerGetAttribute failed (%d)\n", result);

        cudaDeviceSynchronize();
        cuMemFree(ptr);
    }
    else
        printf("cuMemAlloc failed (%d)\n", result);

    cudaDeviceSynchronize();
    cuCtxDestroy(context);
    cudaDeviceReset();
    return 0;
}
