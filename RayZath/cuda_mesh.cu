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
		__host__ CudaMesh::CudaMesh()
			: mesh_structure(nullptr)
		{}

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
				else this->mesh_structure = nullptr;
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
				else ThrowAtCondition(false, L"hMaterial.id out of bounds");
			}
			else ThrowAtCondition(false, L"hMaterial was empty");


			hMesh->GetStateRegister().MakeUnmodified();
		}
		// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	}
}