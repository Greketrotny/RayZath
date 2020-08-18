#ifndef CUDA_ENGINE_KERNEL_CUH
#define CUDA_ENGINE_KERNEL_CUH

#include "cuda_world.cuh"
#include "cuda_render_parts.cuh"

namespace RayZath
{
	namespace CudaKernel
	{
		__global__ void Kernel(
			CudaKernelData* const kernel_data, 
			CudaWorld* const world, 
			const int index);

		void CallKernel();
	}
}

#endif // !CUDA_ENGINE_KERNEL_CUH