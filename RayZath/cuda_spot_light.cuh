#ifndef CUDA_SPOT_LIGHT_CUH
#define CUDA_SPOT_LIGHT_CUH

#include "spot_light.h"
#include "cuda_render_parts.cuh"
#include "world_object.h"
#include "cuda_material.cuh"


namespace RayZath::Cuda
{
	class World;

	class SpotLight
	{
	private:
		vec3f m_position;
		vec3f m_direction;
		float m_size;
		float m_angle, m_cos_angle;
		float m_sharpness;
		ColorF m_color;
		float m_emission;

	public:
		__host__ SpotLight();

	public:
		__host__ void Reconstruct(
			const World& hCudaWorld,
			const RayZath::Engine::Handle<RayZath::Engine::SpotLight>& hSpotLight,
			cudaStream_t& mirror_stream);


		__device__ vec3f GetPosition() const
		{
			return m_position;
		}
		__device__ vec3f GetDirection() const
		{
			return m_direction;
		}
		__device__ float GetSize() const
		{
			return m_size;
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
			const vec3f& point,
			const vec3f& vS,
			float& Se,
			RNG& rng) const
		{
			vec3f vPL;
			float dPL, vOP_dot_vD, dPQ;
			RayPointCalculation(Ray(point, vS), m_position, vPL, dPL, vOP_dot_vD, dPQ);

			if (dPQ < GetSize())
			{	// ray with sample direction would hit the light
				Se = m_emission;
				const float dOQ = sqrtf(dPL * dPL - dPQ * dPQ);
				return vS * dOQ;
			}
			else
			{	// sample random direction on disk
				return SampleDisk(vPL / dPL, m_size, rng) + m_position - point;
			}
		}
		__device__ __inline__ float SolidAngle(const float d) const
		{
			const float A = m_size * m_size * CUDART_PI_F;
			const float d1 = d + 1.0f;
			return A / (d1 * d1);
		}
		__device__ float BeamIllumination(const vec3f& vPL) const
		{
			return float(m_cos_angle < vec3f::Similarity(-vPL, m_direction));
		}
	};
}

#endif // !CUDA_SPOT_LIGHT_CUH