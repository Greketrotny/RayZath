#ifndef CUDA_PLANE_H
#define CUDA_PLANE_H

#include "cuda_render_object.cuh"
#include "plane.h"

namespace RayZath::Cuda
{
	class Plane
		: public RenderObject
	{
	public:
		const Material* material;

		__host__ Plane();

		__host__ void Reconstruct(
			const World& hCudaWorld,
			const RayZath::Engine::Handle<RayZath::Engine::Plane>& hPlane,
			cudaStream_t& mirror_stream);

		__device__ __inline__ bool ClosestIntersection(RayIntersection& intersection) const
		{
			RangedRay objectSpaceRay = intersection.ray;
			transformation.TransformRayG2L(objectSpaceRay);

			const float length_factor = objectSpaceRay.direction.Length();
			objectSpaceRay.near_far *= length_factor;
			objectSpaceRay.direction.Normalize();


			if (objectSpaceRay.direction.y > -1.0e-7f &&
				objectSpaceRay.direction.y < 1.0e-7f) return false;

			const float t = -objectSpaceRay.origin.y / objectSpaceRay.direction.y;
			if (t <= objectSpaceRay.near_far.x || t >= objectSpaceRay.near_far.y) return false;

			intersection.ray.near_far.y = t / length_factor;
			intersection.point = objectSpaceRay.origin + objectSpaceRay.direction * t;
			intersection.texcrd = Texcrd(
				intersection.point.x,
				intersection.point.z);

			// calculate normals
			const float n_factor = float(objectSpaceRay.direction.y < 0.0f) * 2.0f - 1.0f;
			intersection.surface_normal = vec3f(0.0f, 1.0f, 0.0f) * n_factor;

			if (material->GetNormalMap())
			{	// sample normal map
				const ColorF map_color = material->GetNormalMap()->Fetch(intersection.texcrd);
				const vec3f map_normal =
					(vec3f(map_color.red, map_color.green, map_color.blue) *
						2.0f -
						vec3f(1.0f)) *
					n_factor;

				intersection.mapped_normal = map_normal;
				const float temp = intersection.mapped_normal.y;
				intersection.mapped_normal.y = intersection.mapped_normal.z;
				intersection.mapped_normal.z = temp;
			}
			else
			{
				intersection.mapped_normal = intersection.surface_normal;
			}

			// set materials
			intersection.surface_material = this->material;
			intersection.behind_material = nullptr;

			return true;
		}
		__device__ __inline__ ColorF AnyIntersection(const RangedRay& ray) const
		{
			RangedRay objectSpaceRay = ray;
			transformation.TransformRayG2L(objectSpaceRay);

			objectSpaceRay.near_far *= objectSpaceRay.direction.Length();
			objectSpaceRay.direction.Normalize();


			if (objectSpaceRay.direction.y > -1.0e-7f &&
				objectSpaceRay.direction.y < 1.0e-7f) return ColorF(1.0f);

			const float t = -objectSpaceRay.origin.y / objectSpaceRay.direction.y;
			if (t <= objectSpaceRay.near_far.x || t >= objectSpaceRay.near_far.y) return ColorF(1.0f);

			// sample texture for transparency
			const vec3f point = objectSpaceRay.origin + objectSpaceRay.direction * t;
			const Texcrd texcrd(point.x, point.z);
			return material->GetOpacityColor(texcrd);
		}
	};
}

#endif // !CUDA_PLANE_H