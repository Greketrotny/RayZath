#include "cuda_engine_parts.cuh"
#include <stdio.h>

namespace RayZath
{
    namespace CudaKernel
    {
        __global__ void Kernel(cudaVec3<float>* vec)
        {
            *vec /= 2.0f;
        }

        void CallKernel()
        {
            cudaVec3<float>* h_vec = new cudaVec3<float>(0.5f, 0.0f, 5.0f);

            cudaVec3<float>* d_vec = nullptr;
            CudaErrorCheck(cudaMalloc((void**)&d_vec, sizeof(cudaVec3<float>)));

            CudaErrorCheck(cudaMemcpy(d_vec, h_vec, sizeof(cudaVec3<float>), cudaMemcpyHostToDevice));

            Kernel << <1u, 1u >> > (d_vec);

            CudaErrorCheck(cudaDeviceSynchronize());
            CudaErrorCheck(cudaGetLastError());

            CudaErrorCheck(cudaMemcpy(h_vec, d_vec, sizeof(cudaVec3<float>), cudaMemcpyDeviceToHost));

            CudaErrorCheck(cudaFree(d_vec));
            delete h_vec;
        }
    }
}
