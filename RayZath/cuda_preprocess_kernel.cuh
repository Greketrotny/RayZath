#ifndef CUDA_PREPROCESS_KERNEL_CUH
#define CUDA_PREPROCESS_KERNEL_CUH

#include "cuda_world.cuh"

namespace RayZath
{
	namespace CudaEngine
	{
		namespace CudaKernel
		{
			// cumulative samples management
			__global__ void DepthBufferReset(
				CudaWorld* const world,
				const int camera_id);
			__global__ void SpacialReprojection(
				CudaWorld* const world,
				const int camera_id);

			__global__ void CudaCameraUpdateSamplesNumber(
				CudaWorld* const world,
				const int camera_id,
				bool reset_flag);
		}
	}
}

#endif