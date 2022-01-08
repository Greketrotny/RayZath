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
	public:
		vec3f position;
		float size;
		Material material;


	public:
		__host__ PointLight();


	public:
		__host__ void Reconstruct(
			const World& hCudaWorld,
			const RayZath::Engine::Handle<RayZath::Engine::PointLight>& host_light,
			cudaStream_t& mirror_stream);


		__device__ __inline__ bool ClosestIntersection(RayIntersection& intersection) const
		{
			const vec3f vPL = position - intersection.ray.origin;
			const float dPL = vPL.Length();

			// check if light is in ray bounds
			if (dPL <= intersection.ray.near_far.x ||
				dPL >= intersection.ray.near_far.y) return false;
			// check if light is in front of ray
			if (vec3f::DotProduct(vPL, intersection.ray.direction) < 0.0f) return false;

			if (RayToPointDistance(intersection.ray, position) < size)
			{	// ray intersects with the light
				intersection.ray.near_far.y = dPL;
				intersection.surface_material = &material;
				return true;
			}

			return false;
		}
		__device__ __inline__ bool AnyIntersection(const RangedRay& ray) const
		{
			const vec3f vPL = position - ray.origin;
			const float dPL = vPL.Length();

			// check if light is in front of ray
			if (vec3f::DotProduct(vPL, ray.direction) < 0.0f) return false;

			return RayToPointDistance(ray, position) < size;
		}
		__device__ vec3f SampleDirection(
			const vec3f& point,
			const vec3f& vS,
			float& Se,
			RNG& rng) const
		{
			vec3f vPL;
			float dPL, vOP_dot_vD, dPQ;
			RayPointCalculation(Ray(point, vS), position, vPL, dPL, vOP_dot_vD, dPQ);

			if (dPQ < size)
			{	// ray with sample direction would hit the light
				Se = material.GetEmission();
				const float dOQ = sqrtf(dPL * dPL - dPQ * dPQ);
				return vS * dOQ;
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
	};
}

#endif // !CUDA_POINT_LIGHT_H