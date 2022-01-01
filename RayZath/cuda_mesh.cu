#include "cuda_mesh.cuh"

#include "cuda_world.cuh"

namespace RayZath::Cuda
{
	// ~~~~~~~~ [CLASS] MeshStructure ~~~~~~~~
	HostPinnedMemory MeshStructure::m_hpm_trs(sizeof(Triangle) * 64u);
	HostPinnedMemory MeshStructure::m_hpm_nodes(
		sizeof(TreeNode)*
		MeshStructure::sm_max_bvh_depth*
		MeshStructure::sm_max_child_count);

	__host__ MeshStructure::MeshStructure()
		: mp_triangles(nullptr)
		, m_triangle_capacity(0u)
		, m_triangle_count(0u)
		, mp_nodes(nullptr)
		, m_node_capacity(0u)
		, m_node_count(0u)
	{}
	__host__ MeshStructure::~MeshStructure()
	{
		if (mp_triangles)
			CudaErrorCheck(cudaFree(mp_triangles));
		if (mp_nodes)
			CudaErrorCheck(cudaFree(mp_nodes));
	}


	__host__ void MeshStructure::Reconstruct(
		const World& hCudaWorld,
		const RayZath::Engine::Handle<RayZath::Engine::MeshStructure>& hMeshStructure,
		cudaStream_t& mirror_stream)
	{
		if (!hMeshStructure->GetStateRegister().IsModified()) return;

		const uint32_t tree_size = hMeshStructure->GetTriangles().GetBVH().GetTreeSize();
		if (tree_size == 0u || hMeshStructure->GetTriangles().GetCount() == 0u)
		{	// tree is empty so release all content

			if (mp_nodes) CudaErrorCheck(cudaFree(mp_nodes));
			mp_nodes = nullptr;
			m_node_capacity = 0u;
			m_node_count = 0u;

			if (mp_triangles) CudaErrorCheck(cudaFree(mp_triangles));
			mp_triangles = nullptr;
			m_triangle_capacity = 0u;
			m_triangle_count = 0u;

			hMeshStructure->GetStateRegister().MakeUnmodified();
			return;
		}

		// allocate memory for tree nodes and triangles
		if (tree_size != m_node_capacity)
		{
			if (mp_nodes) CudaErrorCheck(cudaFree(mp_nodes));
			m_node_capacity = hMeshStructure->GetTriangles().GetBVH().GetTreeSize();
			CudaErrorCheck(cudaMalloc((void**)&mp_nodes, sizeof(*mp_nodes) * m_node_capacity));
		}
		const uint32_t h_capacity = hMeshStructure->GetTriangles().GetCapacity();
		if (m_triangle_capacity != h_capacity)
		{
			if (mp_triangles) CudaErrorCheck(cudaFree(mp_triangles));
			m_triangle_capacity = h_capacity;
			CudaErrorCheck(cudaMalloc((void**)&mp_triangles, sizeof(*mp_triangles) * m_triangle_capacity));
		}

		m_node_count = 0u;
		m_triangle_count = 0u;

		// reserve hpm for triangle chunks
		const uint32_t trs_chunk_size = uint32_t(m_hpm_trs.GetSize() / sizeof(*mp_triangles));
		Triangle* const hCudaTriangles = (Triangle*)(m_hpm_trs.GetPointerToMemory());
		RZAssert(trs_chunk_size > 16u, "Too few hpm for triangle reconstruction");
		uint32_t trs_in_chunk = 0u;


		auto CopyTrsChunk = [&]() -> void
		{
			if (trs_in_chunk == 0u) return;

			RZAssert(m_triangle_count <= m_triangle_capacity, "qwer");

			CudaErrorCheck(cudaMemcpyAsync(
				mp_triangles + m_triangle_count - trs_in_chunk,
				hCudaTriangles,
				sizeof(*mp_triangles) * trs_in_chunk,
				cudaMemcpyKind::cudaMemcpyHostToDevice, mirror_stream));
			CudaErrorCheck(cudaStreamSynchronize(mirror_stream));
			trs_in_chunk = 0u;
		};
		auto AddTriangle = [&](const RayZath::Engine::Triangle& hTriangle) -> void
		{
			if (trs_in_chunk >= trs_chunk_size)
				CopyTrsChunk();

			Triangle& hCudaTriangle = *(hCudaTriangles + trs_in_chunk);
			new (&hCudaTriangle) Triangle(hTriangle);

			if (hTriangle.AreVertsValid())
			{
				hCudaTriangle.SetVertices(
					vec3f(hMeshStructure->GetVertices()[hTriangle.vertices[0]]),
					vec3f(hMeshStructure->GetVertices()[hTriangle.vertices[1]]),
					vec3f(hMeshStructure->GetVertices()[hTriangle.vertices[2]]));
			}
			else
			{
				hCudaTriangle.SetVertices(
					vec3f(0.0f, 0.0f, 0.0f), vec3f(1.0f, 0.0f, 0.0f), vec3f(0.0f, 1.0f, 0.0f));
			}
			if (hTriangle.AreTexcrdsValid())
			{
				hCudaTriangle.SetTexcrds(
					vec2f(hMeshStructure->GetTexcrds()[hTriangle.texcrds[0]]),
					vec2f(hMeshStructure->GetTexcrds()[hTriangle.texcrds[1]]),
					vec2f(hMeshStructure->GetTexcrds()[hTriangle.texcrds[2]]));
			}
			else
			{
				hCudaTriangle.SetTexcrds(vec2f(), vec2f(), vec2f());
			}
			if (hTriangle.AreNormalsValid())
			{
				hCudaTriangle.SetNormals(
					vec3f(hMeshStructure->GetNormals()[hTriangle.normals[0]]),
					vec3f(hMeshStructure->GetNormals()[hTriangle.normals[1]]),
					vec3f(hMeshStructure->GetNormals()[hTriangle.normals[2]]));
			}
			else
			{
				hCudaTriangle.SetNormals(
					vec3f(hTriangle.normal),
					vec3f(hTriangle.normal),
					vec3f(hTriangle.normal));
			}

			trs_in_chunk++;
			m_triangle_count++;
		};

		auto BuildNode = [&](
			const auto& BuildNodeFunc,
			TreeNode& hCudaNode,
			TreeNode* const hCudaEndNode,
			const RayZath::Engine::ComponentTreeNode<RayZath::Engine::Triangle>& hNode) -> void
		{
			if (hNode.IsLeaf())
			{
				const uint32_t leaf_size = hNode.GetObjectCount();
				hCudaNode.SetRange(m_triangle_count, m_triangle_count + leaf_size);
				for (uint32_t i = 0u; i < leaf_size; i++)
				{
					AddTriangle(*hNode.GetObject(i));
				}
			}
			else
			{
				const uint32_t child_count = hNode.GetChildCount();
				hCudaNode.SetRange(m_node_count, m_node_count + child_count);
				const uint32_t child_begin_idx = m_node_count;
				m_node_count += child_count;
				for (uint32_t i = 0u, c = 0u; i < 8u; i++)
				{
					const auto* const hChildNode =
						hNode.GetChild(i);

					if (hChildNode)
					{
						TreeNode* const hCudaChildNode = hCudaEndNode + c++;
						new (hCudaChildNode) TreeNode(
							hChildNode->GetBoundingBox(),
							hChildNode->IsLeaf());
						BuildNodeFunc(BuildNodeFunc, *hCudaChildNode, hCudaEndNode + child_count, *hChildNode);
					}
				}

				CudaErrorCheck(cudaMemcpyAsync(
					mp_nodes + child_begin_idx,
					hCudaEndNode,
					sizeof(*mp_nodes) * child_count,
					cudaMemcpyKind::cudaMemcpyHostToDevice,
					mirror_stream));
				CudaErrorCheck(cudaStreamSynchronize(mirror_stream));
			}
		};

		// get hpm for root node
		const auto& hRootNode = hMeshStructure->GetTriangles().GetBVH().GetRootNode();
		TreeNode* hCudaRootNode = (TreeNode*)m_hpm_nodes.GetPointerToMemory();
		m_node_count++;

		// build root node
		new (hCudaRootNode) TreeNode(hRootNode.GetBoundingBox(), hRootNode.IsLeaf());
		BuildNode(BuildNode, *hCudaRootNode, hCudaRootNode + 1u, hRootNode);

		// copy root node to device
		CudaErrorCheck(cudaMemcpyAsync(
			mp_nodes,
			hCudaRootNode,
			sizeof(*hCudaRootNode),
			cudaMemcpyKind::cudaMemcpyHostToDevice, mirror_stream));
		CudaErrorCheck(cudaStreamSynchronize(mirror_stream));

		// copy last possibly not  full chunk of triangles to device
		CopyTrsChunk();

		hMeshStructure->GetStateRegister().MakeUnmodified();
	}
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



	// ~~~~~~~~ [CLASS] Mesh ~~~~~~~~
	__host__ Mesh::Mesh()
		: mesh_structure(nullptr)
		, materials{}
	{}

	__host__ void Mesh::Reconstruct(
		const World& hCudaWorld,
		const Engine::Handle<Engine::Mesh>& hMesh,
		cudaStream_t& mirror_stream)
	{
		if (!hMesh->GetStateRegister().IsModified()) return;

		transformation = hMesh->GetTransformation();
		bounding_box = hMesh->GetBoundingBox();

		// mesh structure
		auto& hStructure = hMesh->GetStructure();
		if (hStructure)
		{
			if (hStructure.GetAccessor()->GetIdx() < hCudaWorld.mesh_structures.GetCount())
			{
				this->mesh_structure =
					hCudaWorld.mesh_structures.GetStorageAddress() +
					hStructure.GetAccessor()->GetIdx();
			}
			else this->mesh_structure = nullptr;
		}
		else this->mesh_structure = nullptr;

		// materials
		for (uint32_t i = 0u; i < Engine::Mesh::GetMaterialCapacity(); i++)
		{
			auto& hMaterial = hMesh->GetMaterial(i);
			if (hMaterial)
			{
				if (hMaterial.GetAccessor()->GetIdx() < hCudaWorld.materials.GetCount())
				{
					materials[i] =
						hCudaWorld.materials.GetStorageAddress() +
						hMaterial.GetAccessor()->GetIdx();
				}
				else materials[i] = hCudaWorld.default_material;
			}
			else materials[i] = hCudaWorld.default_material;
		}


		hMesh->GetStateRegister().MakeUnmodified();
	}
}