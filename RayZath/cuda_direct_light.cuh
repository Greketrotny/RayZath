#ifndef CUDA_DIRECT_LIGHT_CUH
#define CUDA_DIRECT_LIGHT_CUH

#include "direct_light.h"
#include "cuda_render_parts.cuh"
#include "cuda_material.cuh"


namespace RayZath
{
	namespace CudaEngine
	{
		class CudaWorld;

		class CudaDirectLight
		{
		public:
			vec3f direction;
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
				const float dot = vec3f::DotProduct(
					intersection.ray.direction,
					-direction);
				if (dot > cos_angular_size)
				{
					intersection.surface_material = &material;
					return true;
				}

				return false;
			}
			__device__ __inline__ vec3f SampleDirection(
				RNG& rng) const
			{
				return SampleSphere(
					rng.UnsignedUniform(),
					rng.UnsignedUniform() * 
					0.5f * (1.0f - cos_angular_size),
					-direction);
			}
		};
	}
}

#endif // !CUDA_DIRECT_LIGHT_CUH