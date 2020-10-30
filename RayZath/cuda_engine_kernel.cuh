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
			//extern __constant__ CudaKernelData c_kernel_data;

			__host__ void CopyToConstantMemory(
				const CudaConstantKernel* hCudaConstantKernel,
				const uint32_t& update_idx,
				cudaStream_t& stream);


			// ~~~~~~~~ Rendering Functions ~~~~~~~~
			// [>] Main Render Pipeline
			__global__ void GenerateCameraRay(
				CudaGlobalKernel* const global_kernel,
				CudaWorld* const world,
				const int camera_id);

			__device__ void TraceRay(
				//const CudaGlobalKernel& kernel,
				ThreadData& thread,
				const CudaWorld& World,
				TracingPath& tracing_path,
				RayIntersection& ray_intersection);
			__device__ CudaColor<float> TraceLightRays(
				//const CudaKernelData& kernel,
				ThreadData& thread,
				const CudaWorld& world,
				RayIntersection& intersection);

			__device__ void GenerateDiffuseRay(
				//const CudaKernelData& kernel,
				ThreadData& thread,
				RayIntersection& intersection);
			__device__ void GenerateSpecularRay(
				//const CudaKernelData& kernel,
				RayIntersection& intersection);
			__device__ void GenerateGlossyRay(
				//const CudaKernelData& kernel,
				ThreadData& thread,
				RayIntersection& intersection);
			__device__ void GenerateTransmissiveRay(
				//const CudaKernelData& kernel,
				ThreadData& thread,
				RayIntersection& intersection);



			// [>] Tone mapping
			__global__ void ToneMap(
				CudaGlobalKernel* const global_kernel,
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