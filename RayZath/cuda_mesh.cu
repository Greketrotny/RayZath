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
		{
			for (size_t i = 0u; i < Mesh::GetMaterialCount(); i++)
				materials[i] = nullptr;
		}

		__host__ void CudaMesh::Reconstruct(
			const CudaWorld& hCudaWorld,
			const Handle<Mesh>& hMesh,
			cudaStream_t& mirror_stream)
		{
			if (!hMesh->GetStateRegister().IsModified()) return;

			// transposition
			this->transformation.position = hMesh->GetPosition();
			this->transformation.rotation = hMesh->GetRotation();
			this->transformation.center = hMesh->GetCenter();
			this->transformation.scale = hMesh->GetScale();
			this->transformation.g2l.ApplyRotationB(-hMesh->GetRotation());
			this->transformation.l2g.ApplyRotation(hMesh->GetRotation());

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

			// materials
			for (uint32_t i = 0u; i < Mesh::GetMaterialCount(); i++)
			{
				auto& hMaterial = hMesh->GetMaterial(i);
				if (hMaterial)
				{
					if (hMaterial.GetResource()->GetId() < hCudaWorld.materials.GetCount())
					{
						materials[i] =
							hCudaWorld.materials.GetStorageAddress() +
							hMaterial.GetResource()->GetId();
					}
					else materials[i] = hCudaWorld.default_material;
				}
				else materials[i] = hCudaWorld.default_material;
			}


			hMesh->GetStateRegister().MakeUnmodified();
		}
		// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	}
}