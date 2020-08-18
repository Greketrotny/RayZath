#include "cuda_sphere.cuh"

namespace RayZath
{
	HostPinnedMemory CudaSphere::m_hpm_CudaTexture(sizeof(CudaTexture));

	__host__ CudaSphere::CudaSphere()
		: radious(1.0f)
		, texture(nullptr)
	{}
	__host__ CudaSphere::~CudaSphere()
	{
		if (this->texture)
		{//--> Destruct texture on device

			CudaTexture* hostCudaTexture = (CudaTexture*)malloc(sizeof(CudaTexture));
			CudaErrorCheck(cudaMemcpy(
				hostCudaTexture, this->texture, 
				sizeof(CudaTexture), 
				cudaMemcpyKind::cudaMemcpyDeviceToHost));

			hostCudaTexture->~CudaTexture();

			CudaErrorCheck(cudaFree(this->texture));
			this->texture = nullptr;

			free(hostCudaTexture);
		}
	}

	__host__ void CudaSphere::Reconstruct(Sphere& hSphere, cudaStream_t& mirror_stream)
	{
		// [>] Update CudaSphere class fields 
		this->position = hSphere.GetPosition();
		this->rotation = hSphere.GetRotation();
		this->scale = hSphere.GetScale();
		this->radious = hSphere.GetRadious();
		this->color = hSphere.GetColor();
		this->material = hSphere.GetMaterial();


		// [>] Mirror CudaSphere class components
		CudaSphere::MirrorTextures(hSphere, mirror_stream);

		hSphere.Updated();
	}
	__host__ void CudaSphere::MirrorTextures(Sphere& hostSphere, cudaStream_t& mirror_stream)
	{
		//if (!hostSphere.UpdateRequests.GetUpdateRequestState(Sphere::SphereUpdateRequestTexture))
		//	return;

		if (hostSphere.GetTexture() != nullptr)
		{// hostSphere has a texture

			if (this->texture == nullptr)
			{// hostCudaSphere doesn't have texture

				// allocate memory for texture
				CudaTexture* hostCudaTexture = (CudaTexture*)malloc(sizeof(CudaTexture));
				// create texture on host
				new (hostCudaTexture) CudaTexture();
				hostCudaTexture->Reconstruct(*hostSphere.GetTexture(), mirror_stream);

				// allocate memory for texture on device
				CudaErrorCheck(cudaMalloc(&this->texture, sizeof(CudaTexture)));
				// copy texture memory to device
				CudaErrorCheck(cudaMemcpy(this->texture, hostCudaTexture, 
					sizeof(CudaTexture), 
					cudaMemcpyKind::cudaMemcpyHostToDevice));
				free(hostCudaTexture);
			}
			else
			{// both sides have texture so do only mirror

				CudaTexture* hCudaTexture = (CudaTexture*)CudaSphere::m_hpm_CudaTexture.GetPointerToMemory();
				//if (CudaSphere::m_hpm_CudaTexture.GetSize() < sizeof(CudaTexture)) return;	// TODO: throw exception (to small memory for mirroring)
				ThrowAtCondition(CudaSphere::m_hpm_CudaTexture.GetSize() >= sizeof(CudaTexture), L"Insufficient host pinned memory for CudaTexture");

				CudaErrorCheck(cudaMemcpyAsync(
					hCudaTexture, this->texture, 
					sizeof(CudaTexture), 
					cudaMemcpyKind::cudaMemcpyDeviceToHost, mirror_stream));
				CudaErrorCheck(cudaStreamSynchronize(mirror_stream));

				hCudaTexture->Reconstruct(*hostSphere.GetTexture(), mirror_stream);

				CudaErrorCheck(cudaMemcpyAsync(
					this->texture, hCudaTexture, 
					sizeof(CudaTexture), 
					cudaMemcpyKind::cudaMemcpyHostToDevice, mirror_stream));
				CudaErrorCheck(cudaStreamSynchronize(mirror_stream));
			}
		}
		else
		{
			if (this->texture != nullptr)
			{// Destroy hostCudaSphere texture

				// host has unloaded texture so destroy texture on device
				CudaTexture* hostCudaTexture = (CudaTexture*)malloc(sizeof(CudaTexture));
				CudaErrorCheck(cudaMemcpy(hostCudaTexture, this->texture, sizeof(CudaTexture), cudaMemcpyKind::cudaMemcpyDeviceToHost));

				hostCudaTexture->~CudaTexture();

				CudaErrorCheck(cudaFree(this->texture));
				this->texture = nullptr;

				free(hostCudaTexture);
			}
		}
	}


	/*__device__ CudaColor<float> CudaSphere::TraceRefractionRay(CudaWorld *world, const CudaRay& ray, Structures::RayIntersection& intersectProps, const int depth)
	{
		//// debug
		//#if defined(__CUDACC__)
		//atomicAdd(&world->debugInfo.rayObjectIntersectTests, 1);
		//#endif

		//if (!(ray.strength * materialFactor > world->renderSettings.materialFactorLowerThreshold &&
		//	depth > 0))
		//{
		//	return this->color;
		//}

		//Structures::RayIntersection internalProps;
		//int internalDepth = depth;

		//// [>] Calculate point of intersection with refraction ray
		//// create objectSpaceRay and transpose it
		//CudaRay objectSpaceRay = ray;
		//objectSpaceRay.origin -= this->position;
		//objectSpaceRay.origin.Rotate<cudaVec3<float>::RotationOrder::ZYX>(-rotation.x, -rotation.y, -rotation.z);
		//objectSpaceRay.direction.Rotate<cudaVec3<float>::RotationOrder::ZYX>(-rotation.x, -rotation.y, -rotation.z);
		//objectSpaceRay.origin /= this->scale;
		//objectSpaceRay.direction /= this->scale;
		//objectSpaceRay.direction.Normalize();

		//do
		//{
		//	// [>] Find point of intersection with internalRay
		//	// calculate scalar t
		//	float tca = -objectSpaceRay.origin.DotProduct(objectSpaceRay.direction);
		//	float d = cudaVec3<float>::DotProduct(objectSpaceRay.origin, objectSpaceRay.origin) - tca * tca;
		//	float delta = radious * radious - d;
		//	float t = tca + sqrtf(delta);

		//	// find point of intersection
		//	internalProps.intersectPoint = objectSpaceRay.origin + objectSpaceRay.direction * t;

		//	// calculate normal
		//	internalProps.surfaceNormal = internalProps.intersectPoint;
		//	internalProps.surfaceNormal /= this->radious;


		//	float dirDotNormal = cudaVec3<float>::Similarity(internalProps.surfaceNormal, objectSpaceRay.direction);
		//	float k = 1.0f - refractiveIndex * refractiveIndex * (1.0f - dirDotNormal * dirDotNormal);

		//	if (k > 0.0f)
		//	{
		//		// [>] Trace ray escaping from the sphere
		//		objectSpaceRay.origin = internalProps.intersectPoint + internalProps.surfaceNormal * 0.001f;
		//		objectSpaceRay.direction = objectSpaceRay.direction * refractiveIndex * (refractiveIndex * dirDotNormal - sqrtf(k));
		//		objectSpaceRay.strength *= materialFactor;

		//		objectSpaceRay.origin *= this->scale;
		//		objectSpaceRay.origin.Rotate(this->rotation);
		//		objectSpaceRay.origin += this->position;

		//		objectSpaceRay.direction /= this->scale;
		//		objectSpaceRay.direction.Rotate(this->rotation);
		//		objectSpaceRay.direction.Normalize();

		//		return CudaEngineKernel::TraceRay(world, objectSpaceRay, intersectProps, internalDepth);
		//
		//		//// compute and trace escaping ray
		//		//refractVector = ray.direction * indexesRatio + reversedNormal * (indexesRatio * dirDotNormal - sqrtf(k));
		//		//CudaRay outRay(P + normal * 0.001f, refractVector, color, ray.strength * materialFactor);
		//		//return CudaEngineKernel::TraceRay(world, outRay, intersectProps, depth - 1);
				//
		//		//intersectProps.surfaceNormal /= this->scale;
		//		//intersectProps.surfaceNormal.Rotate(this->rotation);
		//		//intersectProps.surfaceNormal.Normalize();
				//
		//		//// transpose P to world space
		//		//intersectProps.intersectPoint *= this->scale;
		//		//intersectProps.intersectPoint.Rotate(this->rotation);
		//		//intersectProps.intersectPoint += this->position;
		//
		//	}
		//	else
		//	{
		//		// debug
		//		#if defined(__CUDACC__)
		//		atomicAdd(&world->debugInfo.refractionRays, 1);
		//		#endif

		//		// compute internalRay for total internal reflection
		//		objectSpaceRay.origin = internalProps.intersectPoint;
		//		objectSpaceRay.direction = CudaEngineKernel::ReflectVector(objectSpaceRay.direction, internalProps.surfaceNormal);
		//		//objectSpaceRay.strength *= this->materialFactor;

		//		internalDepth -= 1;
		//	}

		//} while ((objectSpaceRay.strength * materialFactor > world->renderSettings.materialFactorLowerThreshold && internalDepth > 0));


		//return this->color;

		// debug
		#if defined(__CUDACC__)
		atomicAdd(&world->debugInfo.rayObjectIntersectTests, 1);
		#endif


		// [>] Calculate point of intersection with refraction ray
		cudaVec3<float> rayOriginToSphereCenter, normRayDirection;
		cudaVec3<float> P;

		float radiousSqrd = radious * radious;
		float t;

		rayOriginToSphereCenter = position - ray.origin;
		float tca = rayOriginToSphereCenter.DotProduct(ray.direction);
		float d = cudaVec3<float>::DotProduct(rayOriginToSphereCenter, rayOriginToSphereCenter) - tca * tca;
		float thc = sqrtf(radiousSqrd - d);
		t = tca + thc;
		P = ray.origin + ray.direction * t;
		cudaVec3<float> normal = P - position;
		normal.Normalize();



		// [>] Compute refraction ray
		float dirDotNormal = cudaVec3<float>::Similarity(normal, ray.direction);
		float indexesRatio = refractiveIndex;
		float k = 1.0f - indexesRatio * indexesRatio * (1.0f - dirDotNormal * dirDotNormal);

		cudaVec3<float> refractVector;
		CudaColor<float> returnColor = color;
		if (ray.strength * materialFactor > world->renderSettings.materialFactorLowerThreshold &&
			depth > 0)
		{
			cudaVec3<float> reversedNormal = cudaVec3<float>::Reverse(normal);
			if (k < 0.0f)
			{

				// debug
				#if defined(__CUDACC__)
				atomicAdd(&world->debugInfo.refractionRays, 1);
				#endif

				// total internal reflection
				cudaVec3<float> nextInternalVector = reversedNormal * -2.0f * cudaVec3<float>::DotProduct(reversedNormal, ray.direction) + ray.direction;
				CudaRay nextInternalRay(P, nextInternalVector, color, ray.strength * materialFactor);
				return this->TraceRefractionRay(world, nextInternalRay, intersectProps, depth - 1);
			}
			else
			{

				// debug
				#if defined(__CUDACC__)
				atomicAdd(&world->debugInfo.refractionRays, 1);
				#endif

				// compute and trace escaping ray
				refractVector = ray.direction * indexesRatio + reversedNormal * (indexesRatio * dirDotNormal - sqrtf(k));
				CudaRay outRay(P + normal * 0.001f, refractVector, color, ray.strength * materialFactor);
				return CudaEngineKernel::TraceRay(world, outRay, intersectProps, depth - 1);
			}
		}
		else
		{
			return this->color;
		}
	}*/
	/*__device__ CudaColor<float> CudaSphere::TraceTransparentRay(CudaWorld *world, const CudaRay& ray, Structures::RayIntersection& intersectProps, const int depth)
	{
		// [>] Calculate point of intersection with transparent ray
		cudaVec3<float> rayOriginToSphereCenter;
		cudaVec3<float> P;

		float radiousSqrd = radious * radious;
		float t;

		rayOriginToSphereCenter = position - ray.origin;
		float tca = rayOriginToSphereCenter.DotProduct(ray.direction);
		float d = cudaVec3<float>::DotProduct(rayOriginToSphereCenter, rayOriginToSphereCenter) - tca * tca;
		float thc = sqrtf(radiousSqrd - d);
		t = tca + thc;
		P = ray.origin + ray.direction * t * 1.001f;
		cudaVec3<float> normal = P - position;
		normal.Normalize();

		if (ray.strength * materialFactor > world->renderSettings.materialFactorLowerThreshold && depth > 0)
		{
			CudaRay outRay(P, ray.direction, color, ray.strength * materialFactor);
			return CudaEngineKernel::TraceRay(world, outRay, intersectProps, depth - 1);
		}
		else
		{
			return this->color;
		}
	}*/
	// [CLASS] CudaSphere -------------------------|
}