#ifndef CUDA_SPOT_LIGHT_CUH
#define CUDA_SPOT_LIGHT_CUH

#include "spot_light.h"
#include "cuda_render_parts.cuh"
#include "world_object.h"
#include "exist_flag.cuh"


namespace RayZath
{
	namespace CudaEngine
	{
		class CudaWorld;

		class CudaSpotLight : public WithExistFlag
		{
		public:
			cudaVec3<float> position;
			cudaVec3<float> direction;
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
				const cudaVec3<float> vPL = position - intersection.ray.origin;
				const float dPL = vPL.Length();

				if (dPL >= intersection.ray.length) return false;
				const float vPL_dot_vD = cudaVec3<float>::DotProduct(vPL, intersection.ray.direction);
				if (vPL_dot_vD < 0.0f) return false;

				const float dist = RayToPointDistance(intersection.ray, position);
				if (dist < size)
				{
					const float t_dist = sqrtf(
						(size + sharpness) *
						(size + sharpness) -
						dist * dist);

					const cudaVec3<float> test_point =
						intersection.ray.origin + intersection.ray.direction * vPL_dot_vD -
						intersection.ray.direction * t_dist;

					const float LP_dot_D = cudaVec3<float>::Similarity(
						test_point - position, direction);
					if (LP_dot_D > cos_angle)
					{
						intersection.ray.length = dPL;
						intersection.surface_material = &material;
						return true;
					}
				}

				return false;
			}
		};
	}
}

#endif // !CUDA_SPOT_LIGHT_CUH