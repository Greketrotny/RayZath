#ifndef CUDA_ENGINE_KERNEL_CUH
#define CUDA_ENGINE_KERNEL_CUH

#include "cuda_world.cuh"
#include "cuda_render_parts.cuh"

namespace RayZath
{
	namespace CudaKernel
	{
		// [>] Main Render Pipeline
		__global__ void GenerateCameraRay(
			CudaKernelData* const kernel_data, 
			CudaWorld* const world, 
			const int camera_id);

		__device__ void TraceRay(
			CudaKernelData& kernel_data,
			const CudaWorld& World,
			TracingPath& tracing_path,
			RayIntersection& ray_intersection);
		__device__ bool LightsIntersection(
			const CudaWorld& world,
			RayIntersection& intersection);
		__device__ bool ClosestIntersection(
			const CudaWorld& World, 
			RayIntersection& intersection);
		__device__ float AnyIntersection(
			CudaKernelData& kernel_data,
			const CudaWorld& world,
			const CudaRay& shadow_ray);
		__device__ CudaColor<float> TraceLightRays(
			CudaKernelData& kernel_data,
			const CudaWorld& world,
			RayIntersection& intersection);

		__device__ void GenerateDiffuseRay(
			CudaKernelData& kernel,
			RayIntersection& intersection);
		__device__ void GenerateSpecularRay(
			CudaKernelData& kernel,
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

#endif // !CUDA_ENGINE_KERNEL_CUH