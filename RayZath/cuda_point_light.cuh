#ifndef CUDA_POINT_LIGHT_H
#define CUDA_POINT_LIGHT_H

#include "point_light.h"
#include "cuda_render_parts.cuh"
#include "cuda_material.cuh"

namespace RayZath::Cuda
{
	class World;

	class PointLight
	{
	private:
		vec3f m_position;
		float m_size;
		ColorF m_color;
		float m_emission;

	public:
		__host__ PointLight();

	public:
		__host__ void Reconstruct(
			const World& hCudaWorld,
			const RayZath::Engine::Handle<RayZath::Engine::PointLight>& host_light,
			cudaStream_t& mirror_stream);


		__device__ vec3f GetPosition() const
		{
			return m_position;
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

		__device__ vec3f SampleDirection(
			const vec3f& point,
			const vec3f& vS,
			float& Se,
			RNG& rng) const
		{
			vec3f vPL;
			float dPL, vOP_dot_vD, dPQ;
			RayPointCalculation(Ray(point, vS), m_position, vPL, dPL, vOP_dot_vD, dPQ);

			if (dPQ < m_size)
			{	// ray with sample direction would hit the light
				Se = GetEmission();
				const float dOQ = sqrtf(dPL * dPL - dPQ * dPQ);
				return vS * dOQ;
			}
			else
			{	// sample random direction on disk
				return SampleDisk(vPL / dPL, GetSize(), rng) + m_position - point;
			}
		}
		__device__ __inline__ float SolidAngle(const float d) const
		{
			const float A = GetSize() * GetSize() * CUDART_PI_F;
			const float d1 = d + 1.0f;
			return A / (d1 * d1);
		}
	};
}

#endif // !CUDA_POINT_LIGHT_H