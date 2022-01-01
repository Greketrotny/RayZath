#ifndef CUDA_DIRECT_LIGHT_CUH
#define CUDA_DIRECT_LIGHT_CUH

#include "direct_light.h"
#include "cuda_render_parts.cuh"
#include "cuda_material.cuh"


namespace RayZath::Cuda
{
	class World;

	class DirectLight
	{
	public:
		vec3f direction;
		float angular_size;
		float cos_angular_size;
		Material material;


	public:
		__host__ DirectLight();


	public:
		__host__ void Reconstruct(
			const World& hCudaWorld,
			const RayZath::Engine::Handle<RayZath::Engine::DirectLight>& hDirectLight,
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
		__device__ __inline__ float SolidAngle() const
		{
			return 2.0f * CUDART_PI_F * (1.0f - cos_angular_size);
		}
	};
}

#endif // !CUDA_DIRECT_LIGHT_CUH