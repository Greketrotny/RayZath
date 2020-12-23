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
			CudaTexture* texture;
		private:
			static HostPinnedMemory m_hpm_CudaTexture;


		public:
			__host__ CudaSphere();
			__host__ ~CudaSphere();


		public:
			__host__ void Reconstruct(
				const CudaWorld& hCudaWorld, 
				const Handle<Sphere>& hSphere, 
				cudaStream_t& mirror_stream);
		private:
			//__host__ void MirrorTextures(const Handle<Sphere>& hostSphere, cudaStream_t& mirrorStream);


		public:
			__device__ __inline__ bool RayIntersect(RayIntersection& intersection) const
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
				const float d = cudaVec3<float>::DotProduct(objectSpaceRay.origin, objectSpaceRay.origin) - tca * tca;
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
				cudaVec3<float> objectNormal = intersection.point;
				objectNormal /= this->radius;

				// fetch sphere texture
				if (this->texture == nullptr)	intersection.surface_color = this->material->color;
				else intersection.surface_color = this->FetchTexture(objectNormal);


				intersection.surface_normal = intersection.point * n;
				intersection.mapped_normal = intersection.surface_normal;


				const float transmittance =
					(1.0f - intersection.surface_color.alpha) * this->material->transmittance;

				if (tn < 0.0f && transmittance > 0.0f)
				{	// intersection from inside

					// TODO: determine the material behind current material
					// or outer nested material we are currently in.
					// Now assumed to always be air/scene material (default one).
					intersection.material = nullptr;
				}
				else// intersection from outside
				{
					intersection.material = this->material;
				}

				return true;
			}
			__device__ __inline__ float ShadowRayIntersect(const CudaRay& ray) const
			{
				// Points description:
				// O - ray.origin
				// S - sphere center
				// A - closest point to S laying on ray
				// P - intersection point


				// [>] Check trivial ray misses
				cudaVec3<float> vOS = this->position - ray.origin;
				float dOS = vOS.Length();
				float maxASdist = fmaxf(this->scale.x, fmaxf(this->scale.y, this->scale.z)) * this->radius;
				if (dOS - maxASdist >= ray.length)	// sphere is to far from ray origin
					return 1.0f;
				float dOA = cudaVec3<float>::DotProduct(vOS, ray.direction);
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
				objectSpaceRay.length *= objectSpaceRay.direction.Length();
				objectSpaceRay.direction.Normalize();

				float shadow = this->material->transmittance;

				// [>] Find point of intersection
				// calculate scalar t
				float tca = -objectSpaceRay.origin.DotProduct(objectSpaceRay.direction);
				float d = cudaVec3<float>::DotProduct(objectSpaceRay.origin, objectSpaceRay.origin) - tca * tca;
				float delta = radius * radius - d;
				if (delta < 0.0f)	return 1.0f;

				float sqrt_delta = sqrtf(delta);
				float tf = tca + sqrt_delta;
				if (tf <= 0.0f)	return 1.0f;

				float tn = tca - sqrt_delta;
				if (tn > 0.0f)
				{
					// calculate point of intersection in object space
					cudaVec3<float> P = objectSpaceRay.origin + objectSpaceRay.direction * tn;
					cudaVec3<float> vOP = (P - objectSpaceRay.origin);
					if (vOP.Length() > objectSpaceRay.length)	// P is further than ray length
						return 1.0f;


					if (this->texture)
					{
						// calculate object space normal
						cudaVec3<float> objectNormal = P;
						objectNormal /= this->radius;

						const CudaColor<float> color = this->FetchTexture(objectNormal);
						shadow *= (1.0f - color.alpha);
					}
					else
					{
						shadow *= (1.0f - this->material->color.alpha);
					}
					if (shadow < 0.0001f) return shadow;
				}

				// calculate point of intersection in object space
				cudaVec3<float> P = objectSpaceRay.origin + objectSpaceRay.direction * tf;
				cudaVec3<float> vOP = (P - objectSpaceRay.origin);
				if (vOP.Length() > objectSpaceRay.length)	// P is further than ray length
					return 1.0f;


				if (this->texture)
				{
					// calculate object space normal
					cudaVec3<float> objectNormal = P;
					objectNormal /= this->radius;

					const CudaColor<float> color = this->FetchTexture(objectNormal);
					shadow *= (1.0f - color.alpha);
				}
				else
				{
					shadow *= (1.0f - this->material->color.alpha);
				}

				return shadow;
			}
			__device__ __inline__ CudaColor<float> FetchTexture(cudaVec3<float> normal) const
			{
				float u = 0.5f + (atan2f(normal.z, normal.x) / 6.283185f);
				float v = 0.5f - (asinf(normal.y) / 3.141592f);

				float4 color;
				#if defined(__CUDACC__)	
				color = tex2D<float4>(this->texture->textureObject, u, v);
				#endif
				return CudaColor<float>(color.z, color.y, color.x, color.w);
			}
		};
	}
}

#endif // !CUDA_SPHERE_CUH