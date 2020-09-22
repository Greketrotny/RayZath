#ifndef CUDA_BVH_H
#define CUDA_BVH_H

#include "bvh.h"
#include "cuda_object_container.cuh"
#include "cuda_render_parts.cuh"

namespace RayZath
{
	struct CudaTreeNode
	{
	public:
		CudaTreeNode* m_child[8];
		bool m_is_leaf;
		unsigned int m_leaf_first_index, m_leaf_last_index;
		CudaBoundingBox m_bb;


	public:
		__host__ CudaTreeNode()
			: m_is_leaf(true)
			, m_leaf_first_index(0u)
			, m_leaf_last_index(0u)
		{
			for (int i = 0; i < 8; i++) m_child[i] = nullptr;
		}
		template <class HostObject>
		__host__ CudaTreeNode(const TreeNode<HostObject>& hNode)
			: m_is_leaf(hNode.IsLeaf())
			, m_leaf_first_index(0u)
			, m_leaf_last_index(0u)
			, m_bb(hNode.GetBoundingBox())
		{
			for (int i = 0; i < 8; i++) m_child[i] = nullptr;
		}
	};

	template <class HostObject, class CudaObject>
	class CudaBVH
	{
	private:
	public:
		CudaTreeNode* m_nodes;
		unsigned int m_nodes_capacity, m_nodes_count;

		CudaObject** m_ptrs;
		unsigned int m_ptrs_capacity, m_ptrs_count;


	public:
		__host__ CudaBVH()
			: m_nodes(nullptr)
			, m_nodes_capacity(0u)
			, m_nodes_count(0u)
			, m_ptrs(nullptr)
			, m_ptrs_capacity(0u)
			, m_ptrs_count(0u)
		{}
		__host__ ~CudaBVH()
		{
			// delete tree nodes
			if (m_nodes) CudaErrorCheck(cudaFree(m_nodes));
			m_nodes = nullptr;
			m_nodes_capacity = 0u;
			m_nodes_count = 0u;

			// delete objects pointers
			if (m_ptrs) CudaErrorCheck(cudaFree(m_ptrs));
			m_ptrs = nullptr;
			m_ptrs_capacity = 0u;
			m_ptrs_count = 0u;
		}


	public:
		__host__ void Reconstruct(
			ObjectContainerWithBVH<HostObject>& hContainer,
			CudaObjectContainer<HostObject, CudaObject>& hCudaContainer,
			HostPinnedMemory& hpm,
			cudaStream_t& mirror_stream)
		{
			//if (hContainer.GetBVH().GetRootNode() == nullptr) return;	// host bvh is empty

			unsigned int h_tree_size = hContainer.GetBVH().GetTreeSize();

			// [>] Resize capacities
			// resize nodes storage capacity
			if (m_nodes_capacity != h_tree_size)
			{
				m_nodes_capacity = h_tree_size;
				if (m_nodes) CudaErrorCheck(cudaFree(m_nodes));
				CudaErrorCheck(cudaMalloc((void**)&m_nodes, h_tree_size * sizeof(*m_nodes)));
			}

			// resize ptrs storage capacity
			if (m_ptrs_capacity != hContainer.GetCapacity())
			{
				m_ptrs_capacity = hContainer.GetCapacity();
				if (m_ptrs) CudaErrorCheck(cudaFree(m_ptrs));
				CudaErrorCheck(cudaMalloc((void**)&m_ptrs, m_ptrs_capacity * sizeof(*m_ptrs)));
			}

			if (m_ptrs_capacity == 0u || m_nodes_capacity == 0u) return;


			// [>] Allocate host memory
			CudaTreeNode* hCudaTreeNodes = (CudaTreeNode*)malloc(m_nodes_capacity * sizeof(*hCudaTreeNodes));
			CudaObject** hCudaObjectPtrs = (CudaObject**)malloc(m_ptrs_capacity * sizeof(*hCudaObjectPtrs));

			m_nodes_count = 0u;
			m_ptrs_count = 0u;


			// [>] Construct BVH
			new (&hCudaTreeNodes[m_nodes_count]) CudaTreeNode(hContainer.GetBVH().GetRootNode());
			++m_nodes_count;
			FillNode(
				hCudaTreeNodes + m_nodes_count - 1u, hContainer.GetBVH().GetRootNode(), 
				hCudaTreeNodes, hCudaObjectPtrs, 
				hCudaContainer);


			// [>] Copy memory to device
			// copy tree nodes
			CudaErrorCheck(cudaMemcpy(
				m_nodes, hCudaTreeNodes,
				m_nodes_capacity * sizeof(CudaTreeNode),
				cudaMemcpyKind::cudaMemcpyHostToDevice));
			// copy object pointers
			CudaErrorCheck(cudaMemcpy(
				m_ptrs, hCudaObjectPtrs,
				m_ptrs_capacity * sizeof(CudaObject*),
				cudaMemcpyKind::cudaMemcpyHostToDevice));


			// [>] Free host memory
			free(hCudaTreeNodes);
			free(hCudaObjectPtrs);
		}
	private:
		__host__ unsigned int CreateLeaf(unsigned int size)
		{
			if (m_ptrs_count + size > m_ptrs_capacity) return 0u;
			else
			{
				m_ptrs_count += size;
				return m_ptrs_count - size;
			}
		}
		__host__ void FillNode(CudaTreeNode* hCudaNode,
			const TreeNode<HostObject>& hNode,
			CudaTreeNode* hCudaTreeNodes,
			CudaObject** hCudaObjectPtrs,
			CudaObjectContainer<HostObject, CudaObject>& hCudaContainer)
		{
			if (hNode.IsLeaf())
			{
				unsigned int leaf_size = hNode.GetObjectCount();
				hCudaNode->m_leaf_first_index = CreateLeaf(leaf_size);
				hCudaNode->m_leaf_last_index = hCudaNode->m_leaf_first_index + leaf_size;
				for (unsigned int i = 0u; i < leaf_size; i++)
				{
					hCudaObjectPtrs[hCudaNode->m_leaf_first_index + i] = 
						hCudaContainer.GetStorageAddress() + hNode.GetObject(i)->GetId();
				}
			}
			else
			{
				for (int i = 0; i < 8; i++)
				{
					const TreeNode<HostObject>* hChildNode = hNode.GetChild(i);
					if (hChildNode)
					{
						new (&hCudaTreeNodes[m_nodes_count]) CudaTreeNode(*hChildNode);
						++m_nodes_count;

						hCudaNode->m_child[i] = m_nodes + (m_nodes_count - 1u);
						FillNode(
							hCudaTreeNodes + m_nodes_count - 1u, *hChildNode,
							hCudaTreeNodes, hCudaObjectPtrs,
							hCudaContainer);
					}
				}
			}
		}
	};


	template <class HostObject, class CudaObject>
	class CudaObjectContainerWithBVH
	{
	private:
		CudaObjectContainer<HostObject, CudaObject> m_container;
		CudaBVH<HostObject, CudaObject> m_bvh;


	public:
		__host__ CudaObjectContainerWithBVH()
		{

		}
		__host__ ~CudaObjectContainerWithBVH()
		{

		}


	public:
		__host__ void Reconstruct(
			ObjectContainerWithBVH<HostObject>& hContainer,
			HostPinnedMemory& hpm,
			cudaStream_t& mirror_stream)
		{
			hContainer.Update();

			m_container.Reconstruct(hContainer, hpm, mirror_stream);
			m_bvh.Reconstruct(hContainer, m_container, hpm, mirror_stream);
		}

		__device__ __inline__ const CudaObjectContainer<HostObject, CudaObject>& GetContainer() const
		{
			return m_container;
		}
		__device__ __inline__ const CudaBVH<HostObject, CudaObject>& GetBVH() const
		{
			return m_bvh;
		}
	};
}

#endif // !CUDA_BVH_H