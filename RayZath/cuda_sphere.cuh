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
			// Points description:
			// O - ray.origin
			// S - sphere center
			// A - closest point to S laying on ray
			// P - intersection point

			// [>] Check easy ray misses
			cudaVec3<float> vOS = this->position - intersection.ray.origin;
			float dOS = vOS.Magnitude();
			float maxASdist = fmaxf(this->scale.x, fmaxf(this->scale.y, this->scale.z)) * this->radious;
			if (dOS - maxASdist >= intersection.ray.length)	// sphere is to far from ray origin
				return false;
			float dOA = cudaVec3<float>::DotProduct(vOS, intersection.ray.direction);
			float dAS = sqrtf(dOS * dOS - dOA * dOA);
			if (dAS >= maxASdist)	// closest distance is longer than maximum radious
				return false;


			// [>] Transpose objectSpadeRay
			CudaRay objectSpaceRay = intersection.ray;
			objectSpaceRay.origin -= this->position;
			objectSpaceRay.origin.RotateZYX(-rotation);
			objectSpaceRay.direction.RotateZYX(-rotation);
			objectSpaceRay.origin /= this->scale;
			objectSpaceRay.direction /= this->scale;
			float length_factor = objectSpaceRay.direction.Magnitude();
			objectSpaceRay.length *= length_factor;
			objectSpaceRay.direction.Normalize();


			//// [>] Find point of intersection
			//// calculate scalar t
			float tca = -objectSpaceRay.origin.DotProduct(objectSpaceRay.direction);
			float d = cudaVec3<float>::DotProduct(objectSpaceRay.origin, objectSpaceRay.origin) - tca * tca;
			float delta = radious * radious - d;
			if (delta < 0.0f)	return false;
			float sqrt_delta = sqrtf(delta);

			float tf = tca + sqrt_delta;
			if (tf < 0.0f)	return false;

			float t = tf, n = 1.0f;
			float tn = tca - sqrt_delta;
			if (tn < 0.0f) n = -1.0f;
			else t = tn;

			// calculate P
			cudaVec3<float> P = objectSpaceRay.origin + objectSpaceRay.direction * t;

			// check distance to intersection point
			cudaVec3<float> vOP = (P - objectSpaceRay.origin);
			float currDistance = vOP.Magnitude();
			if (currDistance > objectSpaceRay.length) return false;
			else intersection.ray.length = currDistance / length_factor;


			// [>] Fill up intersect properties
			// calculate object space normal
			cudaVec3<float> objectNormal = P;
			objectNormal /= this->radious;

			// fetch sphere texture
			if (this->texture == nullptr)	intersection.surface_color = this->color;
			else intersection.surface_color = this->FetchTexture(objectNormal);

			// calculate world space normal
			intersection.normal = P * n;
			intersection.normal /= this->scale;
			intersection.normal.RotateXYZ(this->rotation);
			intersection.normal.Normalize();

			// calculate world space intersection point
			intersection.point = P;
			intersection.point *= this->scale;
			intersection.point.RotateXYZ(this->rotation);
			intersection.point += this->position;

			return true;
		}
		__device__ __inline__ bool ShadowRayIntersect(const CudaRay& ray) const
		{
			// Points description:
			// O - ray.origin
			// S - sphere center
			// A - closest point to S laying on ray
			// P - intersection point


			// [>] Check trivial ray misses
			cudaVec3<float> OSvec = this->position - ray.origin;
			float OSdist = OSvec.Magnitude();
			float maxASdist = fmaxf(this->scale.x, fmaxf(this->scale.y, this->scale.z)) * this->radious;
			if (OSdist - maxASdist >= ray.length)	// sphere is to far from ray origin
				return false;
			float OAdist = cudaVec3<float>::DotProduct(OSvec, ray.direction);
			float ASdist = sqrtf(OSdist * OSdist - OAdist * OAdist);
			if (ASdist >= maxASdist)	// closest distance is longer than maximum radious
				return false;



			// [>] Transpose objectSpadeRay
			CudaRay objectSpaceRay = ray;
			objectSpaceRay.origin -= this->position;
			objectSpaceRay.origin.RotateZYX(-rotation);
			objectSpaceRay.direction.RotateZYX(-rotation);
			objectSpaceRay.origin /= this->scale;
			objectSpaceRay.direction /= this->scale;
			objectSpaceRay.length *= objectSpaceRay.direction.Magnitude();
			objectSpaceRay.direction.Normalize();


			// [>] Find point of intersection
			// calculate scalar t
			float tca = -objectSpaceRay.origin.DotProduct(objectSpaceRay.direction);
			float d = cudaVec3<float>::DotProduct(objectSpaceRay.origin, objectSpaceRay.origin) - tca * tca;
			float delta = radious * radious - d;
			if (delta < 0.0f)	return false;

			float sqrt_delta = sqrtf(delta);	
			float tf = tca + sqrt_delta;
			if (tf <= 0.0f)	return false;
			float t = tf;
			float tn = tca - sqrt_delta;
			if (tn > 0.0f) t = tn;

			// calculate point of intersection in object space
			cudaVec3<float> objectPoint = objectSpaceRay.origin + objectSpaceRay.direction * t;
			cudaVec3<float> OPvec = (objectPoint - objectSpaceRay.origin) * this->scale;
			if (OPvec.Magnitude() > ray.length)	// P is further than ray length
				return false;

			return true;
		}

		/*__device__ __inline__ void GenerateNextRay(
			CudaEngineKernel::CudaRenderingKernel& renderingKernel,
			CudaEngineKernel::RayIntersection& intersection)
		{
			switch (this->material.type)
			{
				case MaterialType::MaterialTypeDiffuse:
					GenerateDiffuseRay(renderingKernel, intersection);
					break;
				case MaterialType::MaterialTypeSpecular:
					GenerateSpecularRay(intersection);
					break;
			}
		}
		__device__ __inline__ void GenerateDiffuseRay(
			CudaEngineKernel::CudaRenderingKernel& renderingKernel,
			CudaEngineKernel::RayIntersection& intersection)
		{
			cudaVec3<float> sampleDirection;
			CudaEngineKernel::DirectionOnHemisphere(renderingKernel.randomNumbers.GetUnsignedUniform(),
				renderingKernel.randomNumbers.GetUnsignedUniform(),
				intersection.worldNormal, sampleDirection);

			intersection.Reset();
			new (&intersection.worldSpaceRay) CudaRay(intersection.worldPoint + intersection.worldNormal * 0.0001f, sampleDirection);
		}
		__device__ __inline__ void GenerateSpecularRay(CudaEngineKernel::RayIntersection& intersection)
		{
			cudaVec3<float> reflectDir = CudaEngineKernel::ReflectVector(intersection.worldSpaceRay.direction, intersection.worldNormal);
			intersection.Reset();
			new (&intersection.worldSpaceRay) CudaRay(intersection.worldPoint, reflectDir);
		}*/

		//__device__ CudaColor<float> TraceRefractionRay(CudaWorld* world, CudaEngineKernel::RayIntersection& intersection, const int depth);
		//__device__ CudaColor<float> TraceTransparentRay(CudaWorld* world, CudaEngineKernel::RayIntersection& intersection, const int depth);
		__device__ __inline__ CudaColor<float> FetchTexture(cudaVec3<float> normal) const
		{
			float u = 0.5f + (atan2f(normal.z, normal.x) / 6.283185f);
			float v = 0.5f - (asinf(normal.y) / 3.141592f);

			float4 color;
			#if defined(__CUDACC__)	
			color = tex2D<float4>(this->texture->textureObject, u, v);
			#endif
			return CudaColor<float>(color.z, color.y, color.x);
		}
	};
}

#endif // !CUDA_SPHERE_CUH