#ifndef CUDA_POINT_LIGHT_H
#define CUDA_POINT_LIGHT_H

#include "point_light.h"
#include "cuda_render_parts.cuh"
#include "cuda_material.cuh"

namespace RayZath
{
	namespace CudaEngine
	{
		class CudaWorld;

		class CudaPointLight
		{
		public:
			vec3f position;
			float size;
			CudaMaterial material;


		public:
			__host__ CudaPointLight();
			__host__ ~CudaPointLight();


		public:
			__host__ void Reconstruct(
				const CudaWorld& hCudaWorld, 
				const Handle<PointLight>& host_light, 
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
			__device__ __inline__ bool AnyIntersection(const CudaRay& ray) const
			{
				const vec3f vPL = position - ray.origin;
				const float dPL = vPL.Length();

				// check if light is in front of ray
				if (vec3f::DotProduct(vPL, ray.direction) < 0.0f) return false;

				return RayToPointDistance(ray, position) < size;
			}
			__device__ __inline__ vec3f SampleDirection(
				const vec3f& point,
				RNG& rng) const
			{
				const vec3f vN = (point - position).Normalized();
				return SampleDisk(vN, size, rng) + position - point;
			}
		};
	}
}

#endif // !CUDA_POINT_LIGHT_H