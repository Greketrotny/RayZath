#include "cuda_mesh.cuh"

#include "cuda_exception.hpp"
#include "cuda_world.cuh"

namespace RayZath::Cuda
{
	// ~~~~~~~~ [CLASS] MeshStructure ~~~~~~~~
	HostPinnedMemory MeshStructure::m_hpm_trs(sizeof(Triangle) * 1024u);
	HostPinnedMemory MeshStructure::m_hpm_nodes(sizeof(TreeNode) * 1024u);

	__host__ MeshStructure::~MeshStructure()
	{
		if (mp_triangles) RZAssertCoreCUDA(cudaFree(mp_triangles));
		if (mp_nodes) RZAssertCoreCUDA(cudaFree(mp_nodes));
	}

	__host__ void MeshStructure::reconstruct(
		[[maybe_unused]] const World& hCudaWorld,
		const RayZath::Engine::Handle<RayZath::Engine::MeshStructure>& hMeshStructure,
		cudaStream_t& mirror_stream)
	{
		if (!hMeshStructure->stateRegister().IsModified()) return;

		const uint32_t tree_size = hMeshStructure->triangles().getBVH().GetRootNode().treeSize();
		if (tree_size == 0u || hMeshStructure->triangles().count() == 0u)
		{	// tree is empty so release all content

			if (mp_nodes) RZAssertCoreCUDA(cudaFree(mp_nodes));
			mp_nodes = nullptr;
			m_node_capacity = 0u;
			m_node_count = 0u;

			if (mp_triangles) RZAssertCoreCUDA(cudaFree(mp_triangles));
			mp_triangles = nullptr;
			m_triangle_capacity = 0u;
			m_triangle_count = 0u;

			hMeshStructure->stateRegister().MakeUnmodified();
			return;
		}

		// allocate memory for tree nodes and triangles
		if (tree_size != m_node_capacity)
		{
			if (mp_nodes) RZAssertCoreCUDA(cudaFree(mp_nodes));
			m_node_capacity = tree_size;
			RZAssertCoreCUDA(cudaMalloc((void**)&mp_nodes, sizeof(*mp_nodes) * m_node_capacity));
		}
		const uint32_t h_capacity = hMeshStructure->triangles().capacity();
		if (m_triangle_capacity != h_capacity)
		{
			if (mp_triangles) RZAssertCoreCUDA(cudaFree(mp_triangles));
			m_triangle_capacity = h_capacity;
			RZAssertCoreCUDA(cudaMalloc((void**)&mp_triangles, sizeof(*mp_triangles) * m_triangle_capacity));
		}

		m_node_count = 0u;
		m_triangle_count = 0u;

		// reserve hpm for triangle chunks
		const uint32_t trs_chunk_size = uint32_t(m_hpm_trs.size() / sizeof(*mp_triangles));
		Triangle* const hCudaTriangles = (Triangle*)(m_hpm_trs.GetPointerToMemory());
		RZAssert(trs_chunk_size > 16u, "Too few hpm for triangle reconstruction");
		uint32_t trs_in_chunk = 0u;

		const uint32_t nodes_chunk_size = uint32_t(m_hpm_nodes.size() / sizeof(*mp_nodes));
		TreeNode* const h_cuda_nodes = (TreeNode*)(m_hpm_nodes.GetPointerToMemory());
		RZAssert(nodes_chunk_size > 16u, "Too few hpm for tree node reconstruction");
		uint32_t nodes_in_chunk = 0u;


		auto CopyTrianglesChunk = [&]() -> void
		{
			if (trs_in_chunk == 0u) return;

			RZAssert(m_triangle_count <= m_triangle_capacity, "triangle count exceeded capacity");

			RZAssertCoreCUDA(cudaMemcpyAsync(
				mp_triangles + m_triangle_count - trs_in_chunk,
				hCudaTriangles,
				sizeof(*mp_triangles) * trs_in_chunk,
				cudaMemcpyKind::cudaMemcpyHostToDevice, mirror_stream));
			RZAssertCoreCUDA(cudaStreamSynchronize(mirror_stream));
			trs_in_chunk = 0u;
		};
		auto CopyNodesChunk = [&]() -> void
		{
			if (nodes_in_chunk == 0u) return;
			RZAssert(m_node_count <= m_node_capacity, "node count exceeded capacity");

			RZAssertCoreCUDA(cudaMemcpyAsync(
				mp_nodes + m_node_count - nodes_in_chunk,
				h_cuda_nodes,
				sizeof(*mp_nodes) * nodes_in_chunk,
				cudaMemcpyKind::cudaMemcpyHostToDevice, mirror_stream));
			RZAssertCoreCUDA(cudaStreamSynchronize(mirror_stream));
			nodes_in_chunk = 0u;
		};

		auto AddTriangle = [&](const RayZath::Engine::Triangle& hTriangle) -> void
		{
			if (trs_in_chunk >= trs_chunk_size)
				CopyTrianglesChunk();

			Triangle& hCudaTriangle = *(hCudaTriangles + trs_in_chunk);
			new (&hCudaTriangle) Triangle(hTriangle);

			if (hTriangle.areVertsValid())
			{
				hCudaTriangle.setVertices(
					vec3f(hMeshStructure->vertices()[hTriangle.vertices[0]]),
					vec3f(hMeshStructure->vertices()[hTriangle.vertices[1]]),
					vec3f(hMeshStructure->vertices()[hTriangle.vertices[2]]));
			}
			else
			{
				hCudaTriangle.setVertices(
					vec3f(0.0f, 0.0f, 0.0f), vec3f(1.0f, 0.0f, 0.0f), vec3f(0.0f, 1.0f, 0.0f));
			}
			if (hTriangle.areTexcrdsValid())
			{
				hCudaTriangle.setTexcrds(
					vec2f(hMeshStructure->texcrds()[hTriangle.texcrds[0]]),
					vec2f(hMeshStructure->texcrds()[hTriangle.texcrds[1]]),
					vec2f(hMeshStructure->texcrds()[hTriangle.texcrds[2]]));
			}
			else
			{
				hCudaTriangle.setTexcrds(vec2f(), vec2f(), vec2f());
			}
			if (hTriangle.areNormalsValid())
			{
				hCudaTriangle.setNormals(
					vec3f(hMeshStructure->normals()[hTriangle.normals[0]]),
					vec3f(hMeshStructure->normals()[hTriangle.normals[1]]),
					vec3f(hMeshStructure->normals()[hTriangle.normals[2]]));
			}
			else
			{
				hCudaTriangle.setNormals(
					vec3f(hTriangle.normal),
					vec3f(hTriangle.normal),
					vec3f(hTriangle.normal));
			}

			trs_in_chunk++;
			m_triangle_count++;
		};
		auto AddNode = [&](TreeNode&& node) -> void
		{
			if (nodes_in_chunk >= nodes_chunk_size)
				CopyNodesChunk();

			TreeNode& h_cuda_node = *(h_cuda_nodes + nodes_in_chunk);
			h_cuda_node = std::move(node);

			nodes_in_chunk++;
			m_node_count++;
		};

		auto BuildChildren = [&](
			const auto& BuildChildrenFunc,
			const RayZath::Engine::ComponentTreeNode<RayZath::Engine::Triangle>& hNode) -> void
		{
			RZAssert(!hNode.isLeaf(), "node had no children");

			const auto& child1 = hNode.children()->first;
			const auto first_subtree_size = child1.treeSize() - 1;
			if (child1.isLeaf())
			{
				AddNode(TreeNode(
					child1.boundingBox(), 0,
					m_triangle_count, uint32_t(child1.objects().size())));
				for (const auto* object : child1.objects())
					if (object) { AddTriangle(*object); }
			}
			else
			{
				AddNode(TreeNode(
					child1.boundingBox(), uint32_t(child1.children()->type),
					m_node_count + 2, 0));
			}

			const auto& child2 = hNode.children()->second;
			if (child2.isLeaf())
			{
				AddNode(TreeNode(
					child2.boundingBox(), 0,
					m_triangle_count, uint32_t(child2.objects().size())));
				for (const auto* object : child2.objects())
					if (object) { AddTriangle(*object); }
			}
			else
			{
				AddNode(TreeNode(
					child2.boundingBox(), uint32_t(child2.children()->type),
					m_node_count + first_subtree_size + 1, 0));
			}

			if (!child1.isLeaf()) BuildChildrenFunc(BuildChildrenFunc, child1);
			if (!child2.isLeaf()) BuildChildrenFunc(BuildChildrenFunc, child2);
		};

		const auto& hRoot = hMeshStructure->triangles().getBVH().GetRootNode();
		if (hRoot.isLeaf())
		{
			AddNode(TreeNode(
				hRoot.boundingBox(), 0,
				m_triangle_count, uint32_t(hRoot.objects().size())));
			for (const auto* object : hRoot.objects())
				if (object) { AddTriangle(*object); }
		}
		else
		{
			AddNode(TreeNode(
				hRoot.boundingBox(), uint32_t(hRoot.children()->type),
				m_node_count + 1, 0));

			BuildChildren(BuildChildren, hRoot);
		}

		CopyTrianglesChunk();
		CopyNodesChunk();

		hMeshStructure->stateRegister().MakeUnmodified();
	}
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



	// ~~~~~~~~ [CLASS] Mesh ~~~~~~~~
	__host__ Instance::Instance()
		: materials{}
		, m_mesh_idx{}
	{}

	__host__ void Instance::reconstruct(
		const World& hCudaWorld,
		const Engine::Handle<Engine::Instance>& hMesh,
		[[maybe_unused]] cudaStream_t& mirror_stream)
	{
		if (!hMesh || !hMesh->stateRegister().IsModified()) return;

		transformation = hMesh->transformationInGroup();
		bounding_box = hMesh->boundingBox();

		m_mesh_idx = hMesh.accessor()->idx();

		// mesh structure
		auto& hStructure = hMesh->meshStructure();
		if (hStructure)
		{
			if (hStructure.accessor()->idx() < hCudaWorld.mesh_structures.count())
			{
				this->mesh_structure =
					hCudaWorld.mesh_structures.storageAddress() +
					hStructure.accessor()->idx();
			}
			else this->mesh_structure = nullptr;
		}
		else this->mesh_structure = nullptr;

		// materials
		for (uint32_t i = 0u; i < Engine::Instance::materialCapacity(); i++)
		{
			auto& hMaterial = hMesh->material(i);
			if (hMaterial)
			{
				if (hMaterial.accessor()->idx() < hCudaWorld.materials.count())
				{
					materials[i] =
						hCudaWorld.materials.storageAddress() +
						hMaterial.accessor()->idx();
				}
				else materials[i] = hCudaWorld.default_material;
			}
			else materials[i] = hCudaWorld.default_material;
		}


		hMesh->stateRegister().MakeUnmodified();
	}
}