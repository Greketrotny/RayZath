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

				intersection.bvh_factor *= 0.95f;

				// transpose objectSpadeRay
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


				// calculate scalar t
				const float tca = -objectSpaceRay.origin.DotProduct(objectSpaceRay.direction);
				const float d = vec3f::DotProduct(objectSpaceRay.origin, objectSpaceRay.origin) - tca * tca;
				const float delta = radius * radius - d;
				if (delta < 0.0f)	return false;
				const float sqrt_delta = sqrtf(delta);

				const float tf = tca + sqrt_delta;
				if (tf < 0.0f)	return false;

				float t = tf, n = 1.0f;
				const float tn = tca - sqrt_delta;
				if (tn < 0.0f) n = -1.0f;
				else t = tn;


				if (t >= objectSpaceRay.length) return false;

				intersection.point = objectSpaceRay.origin + objectSpaceRay.direction * t;
				intersection.ray.length = t / length_factor;


				// [>] Fill up intersect properties
				// calculate object space normal
				vec3f objectNormal = intersection.point;
				objectNormal /= this->radius;
				intersection.texcrd = CalculateTexcrd(objectNormal);

				intersection.surface_normal = intersection.point * n;
				intersection.mapped_normal = intersection.surface_normal;
				intersection.surface_material = this->material;

				if (tn < 0.0f)
				{	// intersection from inside

					intersection.behind_material = nullptr;
				}
				else
				{	// intersection from outside

					intersection.behind_material = this->material;
				}

				return true;
			}
			__device__ __inline__ float AnyIntersection(const CudaRay& ray) const
			{
				// Points description:
				// O - ray.origin
				// S - sphere center
				// A - closest point to S laying on ray
				// P - intersection point


				// [>] Check trivial ray misses
				vec3f vOS = this->position - ray.origin;
				float dOS = vOS.Length();
				float maxASdist = fmaxf(this->scale.x, fmaxf(this->scale.y, this->scale.z)) * this->radius;
				if (dOS - maxASdist >= ray.length)	// sphere is to far from ray origin
					return 1.0f;
				float dOA = vec3f::DotProduct(vOS, ray.direction);
				float dAS = sqrtf(dOS * dOS - dOA * dOA);
				if (dAS >= maxASdist)	// closest distance is longer than maximum radius
					return 1.0f;



				// [>] Transpose objectSpadeRay
				CudaRay objectSpaceRay = ray;
				objectSpaceRay.origin -= this->position;
				objectSpaceRay.origin.RotateZYX(-rotation);
				objectSpaceRay.direction.RotateZYX(-rotation);
				objectSpaceRay.origin /= this->scale;
				objectSpaceRay.direction /= this->scale;
				objectSpaceRay.origin -= this->center;
				objectSpaceRay.length *= objectSpaceRay.direction.Length();
				objectSpaceRay.direction.Normalize();

				float shadow = this->material->transmittance;

				// [>] Find point of intersection
				// calculate scalar t
				float tca = -objectSpaceRay.origin.DotProduct(objectSpaceRay.direction);
				float d = vec3f::DotProduct(objectSpaceRay.origin, objectSpaceRay.origin) - tca * tca;
				float delta = radius * radius - d;
				if (delta < 0.0f)	return 1.0f;

				float sqrt_delta = sqrtf(delta);
				float tf = tca + sqrt_delta;
				if (tf <= 0.0f)	return 1.0f;

				float tn = tca - sqrt_delta;
				if (tn > 0.0f)
				{
					// calculate point of intersection in object space
					vec3f P = objectSpaceRay.origin + objectSpaceRay.direction * tn;
					vec3f vOP = (P - objectSpaceRay.origin);
					if (vOP.Length() > objectSpaceRay.length)	// P is further than ray length
						return 1.0f;


					// calculate object space normal
					vec3f objectNormal = P;
					objectNormal /= this->radius;
					const CudaColor<float> color = material->GetColor(
						CalculateTexcrd(objectNormal));

					shadow *= (1.0f - color.alpha);
					if (shadow < 0.0001f) return shadow;
				}

				// calculate point of intersection in object space
				vec3f P = objectSpaceRay.origin + objectSpaceRay.direction * tf;
				vec3f vOP = (P - objectSpaceRay.origin);
				if (vOP.Length() > objectSpaceRay.length)	// P is further than ray length
					return 1.0f;


				// calculate object space normal
				vec3f objectNormal = P;
				objectNormal /= this->radius;
				const CudaColor<float> color = material->GetColor(
					CalculateTexcrd(objectNormal));

				shadow *= (1.0f - color.alpha);
				return shadow;
			}
			__device__ __inline__ CudaTexcrd CalculateTexcrd(const vec3f& normal) const
			{
				return CudaTexcrd(
					0.5f + (atan2f(normal.z, normal.x) / 6.283185f),
					0.5f - (asinf(normal.y) / 3.141592f));
			}
		};
	}
}

#endif // !CUDA_SPHERE_CUH