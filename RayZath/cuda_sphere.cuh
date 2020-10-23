#ifndef CUDA_SPHERE_CUH
#define CUDA_SPHERE_CUH

#include "sphere.h"
#include "cuda_render_object.cuh"
#include "cuda_render_parts.cuh"

namespace RayZath
{
	class CudaSphere : public CudaRenderObject
	{
	public:
		CudaColor<float> color;
		float radious;
		CudaTexture* texture;
	private:
		static HostPinnedMemory m_hpm_CudaTexture;


	public:
		__host__ CudaSphere();
		__host__ ~CudaSphere();


	public:
		__host__ void Reconstruct(Sphere& hSphere, cudaStream_t& mirror_stream);
	private:
		__host__ void MirrorTextures(Sphere& hostSphere, cudaStream_t& mirrorStream);


	public:
		__device__ __inline__ bool RayIntersect(RayIntersection& intersection) const
		{
			// [>] check ray intersection with boundingVolume
			if (!boundingVolume.RayIntersection(intersection.ray))
				return false;

			intersection.bvh_factor *= 0.9f;

			// [>] Transpose objectSpadeRay
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


			// [>] Find point of intersection
			// calculate scalar t
			const float tca = -objectSpaceRay.origin.DotProduct(objectSpaceRay.direction);
			const float d = cudaVec3<float>::DotProduct(objectSpaceRay.origin, objectSpaceRay.origin) - tca * tca;
			const float delta = radious * radious - d;
			if (delta < 0.0f)	return false;
			const float sqrt_delta = sqrtf(delta);

			const float tf = tca + sqrt_delta;
			if (tf < 0.0f)	return false;

			float t = tf, n = 1.0f;
			float tn = tca - sqrt_delta;
			if (tn < 0.0f)
			{
				n = -1.0f;
			}
			else t = tn;

			// calculate P
			const cudaVec3<float> P = objectSpaceRay.origin + objectSpaceRay.direction * t;

			// check distance to intersection point
			const cudaVec3<float> vOP = (P - objectSpaceRay.origin);
			const float currDistance = vOP.Length();
			if (currDistance > objectSpaceRay.length) return false;
			else intersection.ray.length = currDistance / length_factor;


			// [>] Fill up intersect properties
			// calculate object space normal
			cudaVec3<float> objectNormal = P;
			objectNormal /= this->radious;

			// fetch sphere texture
			if (this->texture == nullptr)	intersection.surface_color = this->color;
			else intersection.surface_color = this->FetchTexture(objectNormal);
			

			intersection.surface_normal = P * n;
			intersection.mapped_normal = intersection.surface_normal;
			intersection.point = P;


			const float transmitance =
				(1.0f - intersection.surface_color.alpha) * this->material.transmitance;

			if (tn < 0.0f && transmitance > 0.0f)
			{	// intersection from inside

				// TODO: determine the material behind current material
				// or outer nested material we are currently in.
				// Now assumed to always be air/scene material (default one).
				intersection.material = CudaMaterial();
			}
			else// intersection from outside
			{
				intersection.material = this->material;
				intersection.material.transmitance = transmitance;
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
			float maxASdist = fmaxf(this->scale.x, fmaxf(this->scale.y, this->scale.z)) * this->radious;
			if (dOS - maxASdist >= ray.length)	// sphere is to far from ray origin
				return 1.0f;
			float dOA = cudaVec3<float>::DotProduct(vOS, ray.direction);
			float dAS = sqrtf(dOS * dOS - dOA * dOA);
			if (dAS >= maxASdist)	// closest distance is longer than maximum radious
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

			float shadow = this->material.transmitance;

			// [>] Find point of intersection
			// calculate scalar t
			float tca = -objectSpaceRay.origin.DotProduct(objectSpaceRay.direction);
			float d = cudaVec3<float>::DotProduct(objectSpaceRay.origin, objectSpaceRay.origin) - tca * tca;
			float delta = radious * radious - d;
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
					objectNormal /= this->radious;

					const CudaColor<float> color = this->FetchTexture(objectNormal);
					shadow *= (1.0f - color.alpha);
				}
				else
				{
					shadow *= (1.0f - this->color.alpha);
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
				objectNormal /= this->radious;

				const CudaColor<float> color = this->FetchTexture(objectNormal);
				shadow *= (1.0f - color.alpha);
			}
			else
			{
				shadow *= (1.0f - this->color.alpha);
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

#endif // !CUDA_SPHERE_CUH