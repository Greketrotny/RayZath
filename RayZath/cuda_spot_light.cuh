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
	public:
		vec3f position;
		vec3f direction;
		float size;
		float angle, cos_angle;
		float sharpness;
		Material material;


	public:
		__host__ SpotLight();


	public:
		__host__ void Reconstruct(
			const World& hCudaWorld,
			const RayZath::Engine::Handle<RayZath::Engine::SpotLight>& hSpotLight,
			cudaStream_t& mirror_stream);


		__device__ __inline__ bool ClosestIntersection(RayIntersection& intersection) const
		{
			const vec3f vPL = position - intersection.ray.origin;
			const float dPL = vPL.Length();

			if (dPL <= intersection.ray.near_far.x ||
				dPL >= intersection.ray.near_far.y) return false;
			const float vPL_dot_vD = vec3f::DotProduct(vPL, intersection.ray.direction);
			if (vPL_dot_vD < 0.0f) return false;

			const float dist = RayToPointDistance(intersection.ray, position);
			if (dist < size)
			{
				const float t_dist = sqrtf(
					(size + sharpness) *
					(size + sharpness) -
					dist * dist);

				const vec3f test_point =
					intersection.ray.origin + intersection.ray.direction * vPL_dot_vD -
					intersection.ray.direction * t_dist;

				const float LP_dot_D = vec3f::Similarity(
					test_point - position, direction);
				if (LP_dot_D > cos_angle)
				{
					intersection.ray.near_far.y = dPL;
					intersection.surface_material = &material;
					return true;
				}
			}

			return false;
		}
		__device__ __inline__ vec3f SampleDirection(
			const vec3f& point,
			const vec3f& sample_direction,
			float& sample_emission,
			RNG& rng) const
		{
			vec3f vPL;
			float dPL, vOP_dot_vD, dPQ;
			RayPointCalculation(Ray(point, sample_direction), position, vPL, dPL, vOP_dot_vD, dPQ);

			if (dPQ < size)
			{	// ray with sample direction would hit the light
				sample_emission = material.GetEmission();
				const float dOQ = sqrtf(dPL * dPL - dPQ * dPQ);
				return sample_direction * dOQ;
			}
			else
			{	// sample random direction on disk
				return SampleDisk(vPL / dPL, size, rng) + position - point;
			}
		}
		__device__ __inline__ float SolidAngle(const float d) const
		{
			const float A = size * size * CUDART_PI_F;
			const float d1 = d + 1.0f;
			return A / (d1 * d1);
		}
		__device__ bool IntersectsWith(const Ray& ray) const
		{
			return RayToPointDistance(ray, position) < size;
		}
	};
}

#endif // !CUDA_SPOT_LIGHT_CUH