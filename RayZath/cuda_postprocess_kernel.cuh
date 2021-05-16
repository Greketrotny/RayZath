#ifndef CUDA_POSTPROCESS_KERNEL_CUH
#define CUDA_POSTPROCESS_KERNEL_CUH

#include "cuda_render_parts.cuh"
#include "cuda_world.cuh"

namespace RayZath
{
	namespace CudaEngine
	{
		namespace CudaKernel
		{
			/*__global__ void IrradianceReduction(
				CudaGlobalKernel* const global_kernel,
				CudaWorld* const world,
				const int camera_id);*/

			__global__ void SpacialReprojection(
				CudaWorld* const world,
				const int camera_id);
			__global__ void ToneMap(
				CudaGlobalKernel* const global_kernel,
				CudaWorld* const world,
				const int camera_id);
		}
	}
}

#endif // !CUDA_POSTPROCESS_KERNEL_CUH