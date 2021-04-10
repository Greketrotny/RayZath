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


			__device__ __inline__ bool ClosestIntersection(RayIntersection& intersection) const
			{
				CudaRay objectSpaceRay = intersection.ray;
				transformation.TransformRayG2L(objectSpaceRay);

				const float length_factor = objectSpaceRay.direction.Length();
				objectSpaceRay.length *= length_factor;
				objectSpaceRay.direction.Normalize();


				if (objectSpaceRay.direction.y > -1.0e-7f && 
					objectSpaceRay.direction.y < 1.0e-7f) return false;

				const float t = -objectSpaceRay.origin.y / objectSpaceRay.direction.y;
				if (t >= objectSpaceRay.length || t <= 0.0f) return false;

				intersection.ray.length = t / length_factor;
				intersection.point = objectSpaceRay.origin + objectSpaceRay.direction * t;
				intersection.texcrd = CudaTexcrd(
					intersection.point.x,
					intersection.point.z);
				intersection.surface_normal = 
					vec3f(0.0f, 1.0f, 0.0f) * 
					(float(objectSpaceRay.direction.y < 0.0f) * 2.0f - 1.0f);

				intersection.mapped_normal = intersection.surface_normal;
				intersection.surface_material = this->material;
				intersection.behind_material = nullptr;

				return true;
			}
			__device__ __inline__ float AnyIntersection(const CudaRay& ray) const
			{
				CudaRay objectSpaceRay = ray;
				transformation.TransformRayG2L(objectSpaceRay);

				objectSpaceRay.length *= objectSpaceRay.direction.Length();
				objectSpaceRay.direction.Normalize();


				if (objectSpaceRay.direction.y > -1.0e-7f && 
					objectSpaceRay.direction.y < 1.0e-7f) return 1.0f;

				const float t = -objectSpaceRay.origin.y / objectSpaceRay.direction.y;
				if (t >= objectSpaceRay.length || t <= 0.0f) return 1.0f;

				return 0.0f;
			}
		};
	}
}

#endif // !CUDA_PLANE_H