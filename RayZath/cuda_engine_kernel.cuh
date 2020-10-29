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
			// [>] Main Render Pipeline
			__global__ void GenerateCameraRay(
				CudaKernelData* const kernel_data,
				CudaWorld* const world,
				const int camera_id);

			__device__ void TraceRay(
				CudaKernelData& kernel,
				ThreadData& thread,
				const CudaWorld& World,
				TracingPath& tracing_path,
				RayIntersection& ray_intersection);
			__device__ CudaColor<float> TraceLightRays(
				CudaKernelData& kernel,
				ThreadData& thread,
				const CudaWorld& world,
				RayIntersection& intersection);

			__device__ void GenerateDiffuseRay(
				CudaKernelData& kernel,
				ThreadData& thread,
				RayIntersection& intersection);
			__device__ void GenerateSpecularRay(
				CudaKernelData& kernel,
				RayIntersection& intersection);
			__device__ void GenerateGlossyRay(
				CudaKernelData& kernel,
				ThreadData& thread,
				RayIntersection& intersection);
			__device__ void GenerateTransmissiveRay(
				CudaKernelData& kernel,
				ThreadData& thread,
				RayIntersection& intersection);



			// [>] Tone mapping
			__global__ void ToneMap(
				CudaKernelData* const kernel_data,
				CudaWorld* const world,
				const int camera_id);


			// [>] CudaCamera samples management
			__global__ void CudaCameraSampleReset(
				CudaWorld* const world,
				const int camera_id);
			__global__ void CudaCameraUpdateSamplesNumber(
				CudaWorld* const world,
				const int camera_id,
				bool reset_flag);
		}
	}
}

#endif // !CUDA_ENGINE_KERNEL_CUH