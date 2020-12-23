#include "cuda_mesh.cuh"

#include "cuda_texture_types.h"
#include "texture_indirect_functions.h"

#include "cuda_world.cuh"

namespace RayZath
{
	namespace CudaEngine
	{
		// ~~~~~~~~ [CLASS] CudaMeshStructure ~~~~~~~~
		HostPinnedMemory CudaMeshStructure::hostPinnedMemory(0x10000);

		CudaMeshStructure::CudaMeshStructure()
		{}
		CudaMeshStructure::~CudaMeshStructure()
		{}

		void CudaMeshStructure::Reconstruct(
			const CudaWorld& hCudaWorld,
			const Handle<MeshStructure>& hMeshStructure,
			cudaStream_t& mirror_stream)
		{
			if (!hMeshStructure->GetStateRegister().IsModified()) return;

			m_vertices.Reconstruct(hMeshStructure->GetVertices(), hostPinnedMemory, mirror_stream);
			m_texcrds.Reconstruct(hMeshStructure->GetTexcrds(), hostPinnedMemory, mirror_stream);
			m_normals.Reconstruct(hMeshStructure->GetNormals(), hostPinnedMemory, mirror_stream);
			m_triangles.Reconstruct(
				hMeshStructure,
				m_vertices,
				m_texcrds,
				m_normals,
				hostPinnedMemory, mirror_stream);

			hMeshStructure->GetStateRegister().MakeUnmodified();
		}
		// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



		// ~~~~~~~~ [CLASS] CudaMesh ~~~~~~~~
		HostPinnedMemory CudaMesh::hostPinnedMemory(0x10000);

		__host__ CudaMesh::CudaMesh()
			: texture(nullptr)
		{
		}
		__host__ CudaMesh::~CudaMesh()
		{
			// destroy all CudaMesh components
			DestroyTextures();
		}

		__host__ void CudaMesh::Reconstruct(
			const CudaWorld& hCudaWorld,
			const Handle<Mesh>& hMesh,
			cudaStream_t& mirror_stream)
		{
			if (!hMesh->GetStateRegister().IsModified()) return;

			// transposition
			this->position = hMesh->GetPosition();
			this->rotation = hMesh->GetRotation();
			this->center = hMesh->GetCenter();
			this->scale = hMesh->GetScale();

			// bounding box
			this->bounding_box = hMesh->GetBoundingBox();

			// mesh structure
			auto& hStructure = hMesh->GetMeshStructure();
			if (hStructure)
			{
				if (hStructure.GetResource()->GetId() < hCudaWorld.mesh_structures.GetCount())
				{
					this->mesh_structure =
						hCudaWorld.mesh_structures.GetStorageAddress() +
						hStructure.GetResource()->GetId();
				}
			}
			else this->mesh_structure = nullptr;

			// material
			auto& hMaterial = hMesh->GetMaterial();
			if (hMaterial)
			{
				if (hMaterial.GetResource()->GetId() < hCudaWorld.materials.GetCount())
				{
					this->material =
						hCudaWorld.materials.GetStorageAddress() +
						hMaterial.GetResource()->GetId();
				}
			}
			else ThrowAtCondition(false, L"hMaterial.id out of bounds");


			CudaMesh::MirrorTextures(hMesh, &mirror_stream);

			hMesh->GetStateRegister().MakeUnmodified();
		}

		__host__ void CudaMesh::MirrorTextures(
			const Handle<Mesh>& hostMesh, 
			cudaStream_t* mirrorStream)
		{
			return;

			//if (hostMesh.GetTexture() != nullptr)
			//{
			//	if (this->texture == nullptr)
			//	{
			//		// host created texture so device must too
			//		CudaTexture* hostCudaTexture = (CudaTexture*)malloc(sizeof(CudaTexture));
			//		new (hostCudaTexture) CudaTexture();

			//		hostCudaTexture->Reconstruct(*hostMesh.GetTexture(), *mirrorStream);

			//		CudaErrorCheck(cudaMalloc(&this->texture, sizeof(CudaTexture)));
			//		CudaErrorCheck(cudaMemcpy(
			//			this->texture, hostCudaTexture,
			//			sizeof(*this->texture),
			//			cudaMemcpyKind::cudaMemcpyHostToDevice));
			//		free(hostCudaTexture);
			//	}
			//	else
			//	{
			//		//if (!hostMesh.UpdateRequests.GetUpdateRequestState(Mesh::MeshUpdateRequestTexture))
			//		//	return;

			//		// on both sides is texture - only mirror
			//		CudaTexture* hostCudaTexture = (CudaTexture*)this->hostPinnedMemory.GetPointerToMemory();
			//		if (this->hostPinnedMemory.GetSize() < sizeof(CudaTexture)) return;	// TODO: throw an exception (to few host-pinned memory)

			//		CudaErrorCheck(cudaMemcpyAsync(
			//			hostCudaTexture, this->texture,
			//			sizeof(CudaTexture),
			//			cudaMemcpyKind::cudaMemcpyDeviceToHost, *mirrorStream));
			//		CudaErrorCheck(cudaStreamSynchronize(*mirrorStream));

			//		hostCudaTexture->Reconstruct(*hostMesh.GetTexture(), *mirrorStream);

			//		CudaErrorCheck(cudaMemcpy(
			//			this->texture, hostCudaTexture,
			//			sizeof(CudaTexture),
			//			cudaMemcpyKind::cudaMemcpyHostToDevice));
			//		CudaErrorCheck(cudaStreamSynchronize(*mirrorStream));
			//	}
			//}
			//else
			//{
			//	if (this->texture != nullptr)
			//	{
			//		// host has unloaded texture so destroy texture on device
			//		CudaTexture* hostCudaTexture = (CudaTexture*)malloc(sizeof(CudaTexture));
			//		CudaErrorCheck(cudaMemcpy(
			//			hostCudaTexture, this->texture,
			//			sizeof(CudaTexture),
			//			cudaMemcpyKind::cudaMemcpyDeviceToHost));

			//		hostCudaTexture->~CudaTexture();

			//		CudaErrorCheck(cudaFree(this->texture));
			//		this->texture = nullptr;
			//		free(hostCudaTexture);
			//	}
			//}
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
}