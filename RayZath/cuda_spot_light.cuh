#ifndef CUDA_SPOT_LIGHT_CUH
#define CUDA_SPOT_LIGHT_CUH

#include "cuda_render_parts.cuh"
#include "cuda_material.cuh"

#include "spot_light.hpp"
#include "world_object.hpp"


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
		ColorF m_color;
		float m_emission;

	public:
		__host__ SpotLight();

	public:
		__host__ void reconstruct(
			const World& hCudaWorld,
			const RayZath::Engine::Handle<RayZath::Engine::SpotLight>& hSpotLight,
			cudaStream_t& mirror_stream);


		__device__ vec3f position() const
		{
			return m_position;
		}
		__device__ vec3f direction() const
		{
			return m_direction;
		}
		__device__ float size() const
		{
			return m_size;
		}
		__device__ ColorF color() const
		{
			return m_color;
		}
		__device__ float emission() const
		{
			return m_emission;
		}

		__device__ __inline__ vec3f sampleDirection(
			const vec3f& point,
			const vec3f& vS,
			float& Se,
			RNG& rng) const
		{
			vec3f vPL;
			float dPL, vOP_dot_vD, dPQ;
			rayPointCalculation(Ray(point, vS), m_position, vPL, dPL, vOP_dot_vD, dPQ);

			if (dPQ < m_size && vOP_dot_vD > 0.0f)
			{	// ray with sample direction would hit the light
				Se = m_emission;
				const float dOQ = sqrtf(dPL * dPL - dPQ * dPQ);
				return vS * fmaxf(dOQ, 1.0e-4f);
			}
			else
			{	// sample random direction on disk
				return sampleDisk(vPL / dPL, m_size, rng) + m_position - point;
			}
		}
		__device__ __inline__ float solidAngle(const float d) const
		{
			const float A = m_size * m_size * CUDART_PI_F;
			const float d1 = d + 1.0f;
			return A / (d1 * d1);
		}
		__device__ float beamIllumination(const vec3f& vPL) const
		{
			return float(m_cos_angle < vec3f::Similarity(-vPL, m_direction));
		}
	};
}

#endif // !CUDA_SPOT_LIGHT_CUH