#ifndef CUDA_PLANE_H
#define CUDA_PLANE_H

#include "cuda_render_object.cuh"
#include "plane.h"

namespace RayZath
{
	namespace CudaEngine
	{
		class CudaPlane
			: public CudaRenderObject
		{
		public:
			const CudaMaterial* material;

			
			__host__ CudaPlane();


			__host__ void Reconstruct(
				const CudaWorld& hCudaWorld,
				const Handle<Plane>& hPlane,
				cudaStream_t& mirror_stream);


			__device__ __inline__ bool RayIntersect(RayIntersection& intersection) const
			{
				CudaRay objectSpaceRay = intersection.ray;
				objectSpaceRay.origin -= this->position;
				objectSpaceRay.origin.RotateZYX(-rotation);
				objectSpaceRay.direction.RotateZYX(-rotation);
				objectSpaceRay.origin /= this->scale;
				objectSpaceRay.direction /= this->scale;
				objectSpaceRay.origin -= this->center;
				const float length_factor = objectSpaceRay.direction.Length();
				objectSpaceRay.length *= length_factor;
				objectSpaceRay.direction.Normalize();


				const float denom = objectSpaceRay.direction.y;
				if (denom > -1.0e-7f && denom < 1.0e-7f) return false;

				const float t = -objectSpaceRay.origin.y / denom;
				if (t >= objectSpaceRay.length || t <= 0.0f) return false;

				intersection.ray.length = t / length_factor;
				intersection.point = objectSpaceRay.origin + objectSpaceRay.direction * t;
				intersection.texcrd = CudaTexcrd(
					intersection.point.x,
					intersection.point.z);
				intersection.surface_normal = cudaVec3<float>(0.0f, 1.0f, 0.0f);
				intersection.mapped_normal = intersection.surface_normal;
				intersection.surface_material = this->material;
				intersection.behind_material = nullptr;

				return true;
			}
			__device__ __inline__ float ShadowRayIntersect(const CudaRay& ray) const
			{
				return 1.0f;
			}
		};
	}
}

#endif // !CUDA_PLANE_H