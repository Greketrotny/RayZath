#ifndef CUDA_DIRECT_LIGHT_CUH
#define CUDA_DIRECT_LIGHT_CUH

#include "direct_light.hpp"

#include "cuda_render_parts.cuh"
#include "cuda_material.cuh"


namespace RayZath::Cuda
{
	class World;

	class DirectLight
	{
	public:
		vec3f m_direction;
		float m_angular_size;
		float m_cos_angular_size;
		ColorF m_color;
		float m_emission;

	public:
		__host__ DirectLight();

	public:
		__host__ void Reconstruct(
			const World& hCudaWorld,
			const RayZath::Engine::Handle<RayZath::Engine::DirectLight>& hDirectLight,
			cudaStream_t& mirror_stream);

		__device__ vec3f GetDirection() const
		{
			return m_direction;
		}
		__device__ float GetAngularSize() const
		{
			return m_angular_size;
		}
		__device__ float GetCosAngularSize() const
		{
			return m_cos_angular_size;
		}
		__device__ ColorF GetColor() const
		{
			return m_color;
		}
		__device__ float GetEmission() const
		{
			return m_emission;
		}


		__device__ __inline__ vec3f SampleDirection(
			const vec3f& vS,
			float& Se,
			RNG& rng) const
		{
			const float dot = vec3f::DotProduct(vS, -m_direction);
			if (dot > m_cos_angular_size)
			{	// ray with sample direction would hit the light
				Se = m_emission;
				return vS;
			}
			else
			{	// sample random light direction
				return SampleSphere(
					rng.UnsignedUniform(),
					rng.UnsignedUniform() *
					0.5f * (1.0f - m_cos_angular_size),
					-m_direction);
			}
		}
		__device__ __inline__ float SolidAngle() const
		{
			return 2.0f * CUDART_PI_F * (1.0f - m_cos_angular_size);
		}
	};
}

#endif // !CUDA_DIRECT_LIGHT_CUH