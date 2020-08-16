#ifndef CUDA_ENGINE_KERNEL_CUH
#define CUDA_ENGINE_KERNEL_CUH

#include "cuda_engine_parts.cuh"

namespace RayZath
{
	namespace CudaKernel
	{
		__global__ void Kernel(cudaVec3<float>* vec);


		void CallKernel();
	}
}

#endif // !CUDA_ENGINE_KERNEL_CUH