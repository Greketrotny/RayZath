#ifndef CUDA_BVH_H
#define CUDA_BVH_H

#include "bvh.h"
#include "cuda_object_container.cuh"

namespace RayZath
{
	struct CudaTreeNode
	{
	public:
		CudaTreeNode* mp_parent;
		bool m_is_leaf;
		CudaTreeNode* m_child[8];
		unsigned int m_leaf_begin, m_leaf_end;


	public:
		__host__ CudaTreeNode()
			: mp_parent(nullptr)
			, m_is_leaf(true)
			, m_leaf_begin(0u)
			, m_leaf_end(0u)
		{
			for (int i = 0; i < 8; i++) m_child[i] = nullptr;
		}
		__host__ CudaTreeNode(
			CudaTreeNode* parent,
			bool is_leaf)
			: mp_parent(parent)
			, m_is_leaf(is_leaf)
			, m_leaf_begin(0u)
			, m_leaf_end(0u)
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


			// [>] Allocate host memory
			CudaTreeNode* hCudaTreeNodes = (CudaTreeNode*)malloc(m_nodes_capacity * sizeof(*hCudaTreeNodes));
			CudaObject** hCudaObjectPtrs = (CudaObject**)malloc(m_ptrs_capacity * sizeof(*hCudaObjectPtrs));

			m_nodes_count = 0u;
			m_ptrs_count = 0u;


			// [>] Construct BVH
			//InsertNode(hContainer.GetBVH().GetRootNode());

			for (unsigned int i = 0; i < m_nodes_capacity; i++)
			{
				hCudaObjectPtrs[i] = hCudaContainer.GetStorageAddress() + i;
			}
			hCudaObjectPtrs[1] = nullptr;


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
		__host__ void InsertNode(const TreeNode<HostObject>& node)
		{
			if (m_nodes_count >= m_nodes_capacity) return;

			// new (&m_nodes[m_nodes_count]) CudaTree
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
		__device__ __inline__ const CudaBVH<BVH<HostObject>, CudaObject>& GetBVH() const
		{
			return m_bvh;
		}
	};
}

#endif // !CUDA_BVH_H