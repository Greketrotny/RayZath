#ifndef CUDA_SPHERE_CUH
#define CUDA_SPHERE_CUH

#include "sphere.h"
#include "cuda_render_object.cuh"
#include "cuda_render_parts.cuh"

namespace RayZath
{
	namespace CudaEngine
	{
		class CudaWorld;

		class CudaSphere : public CudaRenderObject
		{
		public:
			float radius;
			const CudaMaterial* material;


		public:
			__host__ CudaSphere();


		public:
			__host__ void Reconstruct(
				const CudaWorld& hCudaWorld,
				const Handle<Sphere>& hSphere,
				cudaStream_t& mirror_stream);


		public:
			__device__ __inline__ bool ClosestIntersection(RayIntersection& intersection) const
			{
				// check ray intersection with bounding box
				if (!bounding_box.RayIntersection(intersection.ray))
					return false;

				// transpose objectSpadeRay
				CudaRay objectSpaceRay = intersection.ray;
				transformation.TransformRayG2L(objectSpaceRay);

				const float length_factor = objectSpaceRay.direction.Length();
				objectSpaceRay.near_far *= length_factor;
				objectSpaceRay.direction.Normalize();


				// calculate scalar t
				const float tca = -objectSpaceRay.origin.DotProduct(objectSpaceRay.direction);
				const float d = vec3f::DotProduct(objectSpaceRay.origin, objectSpaceRay.origin) - tca * tca;
				const float delta = radius * radius - d;
				if (delta < 0.0f)	return false;
				const float sqrt_delta = sqrtf(delta);

				const float tf = tca + sqrt_delta;
				if (tf < objectSpaceRay.near_far.x) return false;

				float t = tf, n = 1.0f;
				const float tn = tca - sqrt_delta;
				if (tn < objectSpaceRay.near_far.x) n = -1.0f;
				else t = tn;


				if (t >= objectSpaceRay.near_far.y) return false;
				intersection.ray.near_far.y = t / length_factor;

				const vec3f normal = (objectSpaceRay.origin + objectSpaceRay.direction * t) / this->radius;

				// [>] Fill up intersect properties
				intersection.texcrd = CalculateTexcrd(normal);
				intersection.surface_material = this->material;
				intersection.surface_normal = normal * n;

				// normal mapping
				if (material->GetNormalMap())
				{
					const vec3f tangent = vec3f::CrossProduct(normal, vec3f(0.0f, 1.0f, 0.0f));
					const vec3f bitangent = vec3f::CrossProduct(tangent, normal);

					const ColorF map_color = material->GetNormalMap()->Fetch(intersection.texcrd);
					const vec3f map_normal = vec3f(map_color.red, map_color.green, map_color.blue) * 2.0f - vec3f(1.0f);
					intersection.mapped_normal =
						(normal * map_normal.z + tangent * map_normal.x + bitangent * map_normal.y) * n;
				}
				else
				{
					intersection.mapped_normal = intersection.surface_normal;
				}

				// set behind material
				intersection.behind_material = (tn >= objectSpaceRay.near_far.x ? this->material : nullptr);

				return true;
			}
			__device__ __inline__ ColorF AnyIntersection(const CudaRay& ray) const
			{
				// Points description:
				// O - ray.origin
				// S - sphere center
				// A - closest point to S laying on ray
				// P - intersection point

				ColorF shadow_mask(1.0f);

				// check ray intersection with bounding box
				if (!bounding_box.RayIntersection(ray))
					return shadow_mask;


				// [>] Transform objectSpadeRay
				CudaRay objectSpaceRay = ray;
				transformation.TransformRayG2L(objectSpaceRay);

				objectSpaceRay.near_far *= objectSpaceRay.direction.Length();
				objectSpaceRay.direction.Normalize();


				// [>] Find point of intersection
				// calculate scalar t
				const float tca = -objectSpaceRay.origin.DotProduct(objectSpaceRay.direction);
				const float d = vec3f::DotProduct(objectSpaceRay.origin, objectSpaceRay.origin) - tca * tca;
				const float delta = radius * radius - d;
				if (delta < 0.0f) return shadow_mask;

				const float sqrt_delta = sqrtf(delta);
				const float tf = tca + sqrt_delta;
				if (tf <= objectSpaceRay.near_far.x) return shadow_mask;

				const float tn = tca - sqrt_delta;
				if (tn > objectSpaceRay.near_far.x)
				{
					// calculate point of intersection in object space
					vec3f P = objectSpaceRay.origin + objectSpaceRay.direction * tn;
					vec3f vOP = (P - objectSpaceRay.origin);
					if (vOP.Length() > objectSpaceRay.near_far.y) // P is further than ray length
						return shadow_mask;


					// calculate object space normal
					vec3f objectNormal = P / this->radius;
					// fetch material color
					shadow_mask *= material->GetOpacityColor(
						CalculateTexcrd(objectNormal));
					if (shadow_mask.alpha < 0.0001f) return shadow_mask;
				}

				// calculate point of intersection in object space
				vec3f P = objectSpaceRay.origin + objectSpaceRay.direction * tf;
				vec3f vOP = (P - objectSpaceRay.origin);
				if (vOP.Length() > objectSpaceRay.near_far.y) // P is further than ray length
					return shadow_mask;


				// calculate object space normal
				vec3f objectNormal = P / this->radius;
				// fetch material color
				shadow_mask *= material->GetOpacityColor(
					CalculateTexcrd(objectNormal));
				return shadow_mask;
			}
			__device__ __inline__ CudaTexcrd CalculateTexcrd(const vec3f& normal) const
			{
				return CudaTexcrd(
					0.5f + (atan2f(normal.z, normal.x) / (2.0f * CUDART_PI_F)),
					0.5f + (asinf(normal.y) / CUDART_PI_F));
			}
		};
	}
}

#endif // !CUDA_SPHERE_CUH