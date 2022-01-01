#ifndef CUDA_SPOT_LIGHT_CUH
#define CUDA_SPOT_LIGHT_CUH

#include "spot_light.h"
#include "cuda_render_parts.cuh"
#include "world_object.h"
#include "cuda_material.cuh"


namespace RayZath
{
	namespace CudaEngine
	{
		class CudaWorld;

		class CudaSpotLight
		{
		public:
			vec3f position;
			vec3f direction;
			float size;
			float angle, cos_angle;
			float sharpness;
			CudaMaterial material;


		public:
			__host__ CudaSpotLight();
			__host__ ~CudaSpotLight();


		public:
			__host__ void Reconstruct(
				const CudaWorld& hCudaWorld, 
				const Handle<SpotLight>& hSpotLight, 
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
				RNG& rng) const
			{
				const vec3f vN = (point - position).Normalized();
				return SampleDisk(vN, size, rng) + position - point;
			}
			__device__ __inline__ float SolidAngle(const float d) const
			{
				const float A = size * size * CUDART_PI_F;
				const float d1 = d + 1.0f;
				return A / (d1 * d1);
			}
		};
	}
}

#endif // !CUDA_SPOT_LIGHT_CUH