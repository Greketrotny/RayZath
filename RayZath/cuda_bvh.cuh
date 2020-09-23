#ifndef CUDA_BVH_H
#define CUDA_BVH_H

#include "bvh.h"
#include "cuda_object_container.cuh"
#include "cuda_render_parts.cuh"

#include "cuda_render_object.cuh"

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

	struct TraversalStack
	{
	public:
		static constexpr unsigned int s_stack_max_size = 8u;
	private:
		CudaTreeNode* m_node[s_stack_max_size];
		char m_depth;
		unsigned char m_start_node;
		unsigned int m_child_counters;


	public:
		__device__ TraversalStack(
			CudaTreeNode* root, const cudaVec3<float>& ray_direction)
			: m_depth(0)
			, m_child_counters(0u)
		{
			m_node[0] = root;
			m_start_node =
				(unsigned int(ray_direction.x > 0.0f) << 2u) |
				(unsigned int(ray_direction.y > 0.0f) << 1u) |
				(unsigned int(ray_direction.z > 0.0f));
		}
		__device__ CudaTreeNode*& GetCurrentNode()
		{
			return m_node[m_depth];
		}
		__device__ CudaTreeNode*& GetChildNode()
		{
			//return m_node[m_depth]->m_child[(m_child_ids >> (4u * m_depth)) & 0b111u];
			return m_node[m_depth]->m_child[((m_child_counters >> (4u * m_depth)) & 0b111u) ^ m_start_node];
		}
		__device__ const char& GetDepth()
		{
			return m_depth;
		}

		__device__ unsigned int GetCheckedCount()
		{
			return ((m_child_counters >> (4u * m_depth)) & 0b1111u);
		}
		__device__ void ResetChildId()
		{
			m_child_counters &= (~(0b1111u << (4u * unsigned int(m_depth))));
		}
		__device__ void IncrementChildId()
		{
			m_child_counters += (1u << (4u * m_depth));
		}

		__device__ void IncrementDepth()
		{
			++m_depth;
		}
		__device__ void DecrementDepth()
		{
			--m_depth;
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


	public:
		__device__ __inline__ void Traverse(
			RayIntersection& intersection,
			const CudaRenderObject*& closest_object) const
		{
			if (m_nodes_count == 0u) return;

			TraversalStack stack(&m_nodes[0], intersection.ray.direction);

			if (stack.GetCurrentNode()->m_bb.RayIntersection(intersection.ray))
			{
				intersection.bvh_factor *= 0.95f;

				while (stack.GetDepth() >= 0 && stack.GetDepth() < TraversalStack::s_stack_max_size - 1u)
				{
					if (stack.GetCurrentNode()->m_is_leaf)
					{
						for (unsigned int i = stack.GetCurrentNode()->m_leaf_first_index;
							i < stack.GetCurrentNode()->m_leaf_last_index;
							i++)
						{
							const CudaObject* object = m_ptrs[i];
							if (object->RayIntersect(intersection))
							{
								closest_object = object;
							}
						}
						stack.DecrementDepth();
					}
					else
					{
						if (stack.GetCheckedCount() >= 8u)
						{
							stack.DecrementDepth();
						}
						else
						{
							CudaTreeNode* child_node = stack.GetChildNode();
							stack.IncrementChildId();

							if (child_node)
							{
								if (child_node->m_bb.RayIntersection(intersection.ray))
								{
									intersection.bvh_factor *= 0.1f * float(stack.GetCheckedCount());

									stack.IncrementDepth();
									stack.GetCurrentNode() = child_node;
									stack.ResetChildId();
								}
							}
						}
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