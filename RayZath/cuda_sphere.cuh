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
		cudaVec3<float> scale;
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
			cudaVec3<float> vOS = this->position - intersection.worldSpaceRay.origin;
			float dOS = vOS.Magnitude();
			float maxASdist = fmaxf(this->scale.x, fmaxf(this->scale.y, this->scale.z)) * this->radious;
			if (dOS - maxASdist >= intersection.worldSpaceRay.length)	// sphere is to far from ray origin
				return false;
			float dOA = cudaVec3<float>::DotProduct(vOS, intersection.worldSpaceRay.direction);
			float dAS = sqrtf(dOS * dOS - dOA * dOA);
			if (dAS >= maxASdist)	// closest distance is longer than maximum radious
				return false;


			// [>] Transpose objectSpadeRay
			intersection.objectSpaceRay = intersection.worldSpaceRay;
			intersection.objectSpaceRay.origin -= this->position;
			intersection.objectSpaceRay.origin.RotateZYX(-rotation);
			intersection.objectSpaceRay.direction.RotateZYX(-rotation);
			intersection.objectSpaceRay.origin /= this->scale;
			intersection.objectSpaceRay.direction /= this->scale;
			intersection.objectSpaceRay.direction.Normalize();


			// [>] Find point of intersection
			// calculate scalar t
			float tca = -intersection.objectSpaceRay.origin.DotProduct(intersection.objectSpaceRay.direction);
			float d = cudaVec3<float>::DotProduct(intersection.objectSpaceRay.origin, intersection.objectSpaceRay.origin) - tca * tca;
			float delta = radious * radious - d;
			if (delta < 0.0f)	return false;
			float t = tca - sqrtf(delta);
			if (t < 0.0f)	return false;

			// find point of intersection
			intersection.objectPoint = intersection.objectSpaceRay.origin + intersection.objectSpaceRay.direction * t;

			// check distance to intersection point
			cudaVec3<float> OPvec = (intersection.objectPoint - intersection.objectSpaceRay.origin) * this->scale;
			float currDistance = OPvec.Magnitude();
			if (currDistance > intersection.worldSpaceRay.length) return false;
			else intersection.worldSpaceRay.length = currDistance;


			// [>] Fill up intersect properties
			// calculate object space normal
			intersection.objectNormal = intersection.objectPoint;
			intersection.objectNormal /= this->radious;

			// fetch sphere texture
			if (this->texture == nullptr)	intersection.surfaceColor = this->color;
			else intersection.surfaceColor = this->FetchTexture(intersection.objectNormal);

			// calculate world space normal
			intersection.worldNormal = intersection.objectNormal;
			intersection.worldNormal /= this->scale;
			intersection.worldNormal.RotateXYZ(this->rotation);
			intersection.worldNormal.Normalize();

			// calculate world space intersection point
			intersection.worldPoint = intersection.objectPoint;
			intersection.worldPoint *= this->scale;
			intersection.worldPoint.RotateXYZ(this->rotation);
			intersection.worldPoint += this->position;

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
			objectSpaceRay.direction.Normalize();


			// [>] Find point of intersection
			// calculate scalar t
			float tca = -objectSpaceRay.origin.DotProduct(objectSpaceRay.direction);
			float d = cudaVec3<float>::DotProduct(objectSpaceRay.origin, objectSpaceRay.origin) - tca * tca;
			float delta = radious * radious - d;
			if (delta < 0.0f)	return false;
			float t = tca - sqrtf(delta);
			if (t < 0.0f)	return false;

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