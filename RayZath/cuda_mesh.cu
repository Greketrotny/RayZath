#include "cuda_mesh.cuh"

#include "cuda_texture_types.h"
#include "texture_indirect_functions.h"

namespace RayZath
{
	// ~~~~~~~~ [CLASS] CudaMesh ~~~~~~~~
	HostPinnedMemory CudaMesh::hostPinnedMemory(0xFFFF);

	__host__ CudaMesh::CudaMesh()
		: vertices()
		, texcrds()
		, triangles()
		, texture(nullptr)
	{
	}
	__host__ CudaMesh::~CudaMesh()
	{
		// destroy all CudaMesh components
		DestroyTextures();
	}

	__host__ void CudaMesh::Reconstruct(
		Mesh& hMesh, 
		cudaStream_t& mirror_stream)
	{
		if (!hMesh.GetStateRegister().IsModified()) return;

		this->vertices.Reconstruct(hMesh.GetMeshStructure().GetVertices(), hostPinnedMemory, mirror_stream);
		this->texcrds.Reconstruct(hMesh.GetMeshStructure().GetTexcrds(), hostPinnedMemory, mirror_stream);
		this->triangles.Reconstruct(
			hMesh.GetMeshStructure().GetTriangles(),
			hMesh.GetMeshStructure().GetVertices(),
			hMesh.GetMeshStructure().GetTexcrds(),
			this->vertices,
			this->texcrds,
			hostPinnedMemory, mirror_stream);

		this->position = hMesh.GetPosition();
		this->rotation = hMesh.GetRotation();
		this->center = hMesh.GetCenter();
		this->scale = hMesh.GetScale();
		this->material = hMesh.GetMaterial();
		this->boundingVolume = hMesh.GetBoundingBox();

		CudaMesh::MirrorTextures(hMesh, &mirror_stream);

		hMesh.GetStateRegister().MakeUnmodified();
	}

	__host__ void CudaMesh::MirrorTextures(const Mesh& hostMesh, cudaStream_t* mirrorStream)
	{
		if (hostMesh.GetTexture() != nullptr)
		{
			if (this->texture == nullptr)
			{
				// host created texture so device must too
				CudaTexture* hostCudaTexture = (CudaTexture*)malloc(sizeof(CudaTexture));
				new (hostCudaTexture) CudaTexture();

				hostCudaTexture->Reconstruct(*hostMesh.GetTexture(), *mirrorStream);

				CudaErrorCheck(cudaMalloc(&this->texture, sizeof(CudaTexture)));
				CudaErrorCheck(cudaMemcpy(
					this->texture, hostCudaTexture, 
					sizeof(*this->texture), 
					cudaMemcpyKind::cudaMemcpyHostToDevice));
				free(hostCudaTexture);
			}
			else
			{
				//if (!hostMesh.UpdateRequests.GetUpdateRequestState(Mesh::MeshUpdateRequestTexture))
				//	return;

				// on both sides is texture - only mirror
				CudaTexture* hostCudaTexture = (CudaTexture*)this->hostPinnedMemory.GetPointerToMemory();
				if (this->hostPinnedMemory.GetSize() < sizeof(CudaTexture)) return;	// TODO: throw an exception (to few host-pinned memory)

				CudaErrorCheck(cudaMemcpyAsync(
					hostCudaTexture, this->texture, 
					sizeof(CudaTexture), 
					cudaMemcpyKind::cudaMemcpyDeviceToHost, *mirrorStream));
				CudaErrorCheck(cudaStreamSynchronize(*mirrorStream));

				hostCudaTexture->Reconstruct(*hostMesh.GetTexture(), *mirrorStream);

				CudaErrorCheck(cudaMemcpy(
					this->texture, hostCudaTexture, 
					sizeof(CudaTexture), 
					cudaMemcpyKind::cudaMemcpyHostToDevice));
				CudaErrorCheck(cudaStreamSynchronize(*mirrorStream));
			}
		}
		else
		{
			if (this->texture != nullptr)
			{
				// host has unloaded texture so destroy texture on device
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
	}
	__host__ void CudaMesh::DestroyTextures()
	{
		if (this->texture)
		{
			CudaTexture* hostCudaTexture = (CudaTexture*)malloc(sizeof(CudaTexture));
			CudaErrorCheck(cudaMemcpy(
				hostCudaTexture, this->texture, 
				sizeof(CudaTexture), 
				cudaMemcpyKind::cudaMemcpyDeviceToHost));

			hostCudaTexture->~CudaTexture();

			free(hostCudaTexture);

			CudaErrorCheck(cudaFree(this->texture));
			this->texture = nullptr;
		}
	}
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
}