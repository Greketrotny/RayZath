#ifndef CUDA_DIRECT_LIGHT_CUH
#define CUDA_DIRECT_LIGHT_CUH

#include "direct_light.h"
#include "cuda_render_parts.cuh"
#include "exist_flag.cuh"
#include "cuda_material.cuh"

namespace RayZath
{
	namespace CudaEngine
	{
		class CudaWorld;

		class CudaDirectLight : public WithExistFlag
		{
		public:
			cudaVec3<float> direction;
			float angular_size;
			float cos_angular_size;
			CudaMaterial material;


		public:
			__host__ CudaDirectLight();
			__host__ ~CudaDirectLight();


		public:
			__host__ void Reconstruct(
				const CudaWorld& hCudaWorld,
				const Handle<DirectLight>& hDirectLight,
				cudaStream_t& mirror_stream);


			__device__ __inline__ bool ClosestIntersection(RayIntersection& intersection) const
			{
				const float dot = cudaVec3<float>::DotProduct(
					intersection.ray.direction,
					-direction);
				if (dot > cos_angular_size)
				{
					intersection.surface_material = &material;
					return true;
				}

				return false;
			}
		};
	}
}

#endif // !CUDA_DIRECT_LIGHT_CUH