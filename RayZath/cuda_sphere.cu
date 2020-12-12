#include "cuda_sphere.cuh"

namespace RayZath
{
	namespace CudaEngine
	{
		HostPinnedMemory CudaSphere::m_hpm_CudaTexture(sizeof(CudaTexture));

		__host__ CudaSphere::CudaSphere()
			: radius(1.0f)
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
			if (!hSphere.GetStateRegister().IsModified()) return;

			// [>] Update CudaSphere class fields 
			this->position = hSphere.GetPosition();
			this->rotation = hSphere.GetRotation();
			this->scale = hSphere.GetScale();
			this->radius = hSphere.GetRadius();
			//this->material = *hSphere.GetMaterial();
			material.Reconstruct(hSphere.GetMaterial(), mirror_stream);
			this->bounding_box = hSphere.GetBoundingBox();


			// [>] Mirror CudaSphere class components
			CudaSphere::MirrorTextures(hSphere, mirror_stream);

			hSphere.GetStateRegister().MakeUnmodified();
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
	}
}