#ifndef CUDA_ENGINE_KERNEL_CUH
#define CUDA_ENGINE_KERNEL_CUH

#include "cuda_world.cuh"
#include "cuda_render_parts.cuh"

namespace RayZath
{
	namespace CudaEngine
	{
		namespace CudaKernel
		{
			// ~~~~~~~~ Memory Management ~~~~~~~~
			__host__ void CopyToConstantMemory(
				const CudaConstantKernel* hCudaConstantKernel,
				const uint32_t& update_idx,
				cudaStream_t& stream);


			// ~~~~~~~~ Rendering Functions ~~~~~~~~
			__global__ void GenerateCameraRay(
				CudaGlobalKernel* const global_kernel,
				CudaWorld* const world,
				const int camera_id);

			__device__ void TraceRay(
				ThreadData& thread,
				const CudaWorld& World,
				TracingPath& tracing_path,
				RayIntersection& ray_intersection);

			__device__ 

			__device__ CudaColor<float> SurfaceDirectSampling(
				ThreadData& thread,
				const CudaWorld& world,
				RayIntersection& intersection);
			__device__ CudaColor<float> PointDirectSampling(
				ThreadData& thread,
				const CudaWorld& world,
				RayIntersection& intersection);
		}
	}
}

#endif // !CUDA_ENGINE_KERNEL_CUH