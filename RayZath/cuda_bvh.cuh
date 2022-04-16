#ifndef CUDA_BVH_H
#define CUDA_BVH_H

#include "bvh.h"
#include "cuda_object_container.cuh"
#include "cuda_render_parts.cuh"

#include "cuda_bvh_tree_node.cuh"

#include <memory>

namespace RayZath::Cuda
{
	template <class HostObject, class CudaObject>
	class ObjectContainerWithBVH
	{
	private:
		ObjectContainer<HostObject, CudaObject> m_container;

		TreeNode* m_nodes = nullptr;
		uint32_t m_capacity = 0, m_count = 0;

	public:
		__host__ ~ObjectContainerWithBVH()
		{
			if (m_nodes) CudaErrorCheck(cudaFree(m_nodes));
		}

	public:
		__host__ void Reconstruct(
			const World& hCudaWorld,
			RayZath::Engine::ObjectContainerWithBVH<HostObject>& hContainer,
			HostPinnedMemory& hpm,
			cudaStream_t& mirror_stream)
		{
			if (!hContainer.GetStateRegister().IsModified()) return;

			const auto tree_size = hContainer.root().treeSize();
			if (tree_size == 0u)
			{
				if (m_nodes) CudaErrorCheck(cudaFree(m_nodes));
				m_nodes = nullptr;
				m_capacity = 0u;
				m_count = 0u;
				hContainer.GetStateRegister().MakeUnmodified();
				return;
			}

			// allocate device memory for nodes
			if (m_capacity != tree_size)
			{
				m_capacity = tree_size;
				if (m_nodes) CudaErrorCheck(cudaFree(m_nodes));
				CudaErrorCheck(cudaMalloc((void**)&m_nodes, sizeof(*m_nodes) * m_capacity));
			}

			// make every object modified, because bvh indexes objects in memory
			for (uint32_t i = 0; i < hContainer.GetCount(); i++)
				hContainer[i]->GetStateRegister().MakeModified();

			// allocate host memory to construct nodes
			auto hCudaTreeNodes = std::make_unique<TreeNode[]>(tree_size);

			// construct BVH
			std::vector<uint32_t> reordered_ids;
			m_count = 1;
			ConstructNode(
				hCudaTreeNodes[0],
				hCudaTreeNodes.get(),
				hContainer.root(),
				reordered_ids);

			RZAssert(m_capacity == m_count, "capacity is set to tree size so should match node count");

			// copy tree nodes constructed on host to device memory
			CudaErrorCheck(cudaMemcpy(
				m_nodes, hCudaTreeNodes.get(),
				m_capacity * sizeof(TreeNode),
				cudaMemcpyKind::cudaMemcpyHostToDevice));

			// reconstruct linear container along reordered list of indices
			m_container.Reconstruct(hCudaWorld, hContainer, reordered_ids, hpm, mirror_stream);

			hContainer.GetStateRegister().MakeUnmodified();
		}
		__host__ void ConstructNode(
			TreeNode& hCudaNode,
			TreeNode* const hCudaNodes,
			const RayZath::Engine::TreeNode<HostObject>& hNode,
			std::vector<uint32_t>& reordered_ids)
		{
			if (hNode.isLeaf())
			{
				hCudaNode = TreeNode(
					hNode.boundingBox(), 0,
					uint32_t(reordered_ids.size()), uint32_t(hNode.objects().size()));
				for (auto& object : hNode.objects())
					reordered_ids.push_back(object.GetAccessor()->GetIdx());
			}
			else
			{
				hCudaNode = TreeNode(
					hNode.boundingBox(), uint32_t(hNode.children()->type),
					m_count, 0);

				const auto first_child_idx = m_count++;
				const auto second_child_idx = m_count++;
				ConstructNode(hCudaNodes[first_child_idx], hCudaNodes, hNode.children()->first, reordered_ids);
				ConstructNode(hCudaNodes[second_child_idx], hCudaNodes, hNode.children()->second, reordered_ids);
			}
		}

	public:
		__device__ __inline__ void ClosestIntersection(RangedRay& ray, TraversalResult& traversal) const
		{
			if (m_count == 0u) return;	// the tree is empty
			if (!m_nodes[0].intersectsWith(ray)) return;	// ray misses root node

			// single node shortcut
			if (m_nodes[0].isLeaf())
			{
				// check all objects held by the node
				for (uint32_t i = m_nodes[0].begin(); i < m_nodes[0].end(); i++)
					m_container[i].ClosestIntersection(ray, traversal);
				return;
			}

			// start node index (bit set means, this axis has flipped traversal order)
			const uint8_t start_node =
				(uint8_t(ray.direction.x < 0.0f) << 2u) |
				(uint8_t(ray.direction.y < 0.0f) << 1u) |
				(uint8_t(ray.direction.z < 0.0f));
			uint8_t depth = 1;	// current depth
			uint32_t node_idx[32u];	// nodes in stack
			node_idx[depth] = 0u;
			uint32_t child_counters = 0u;

			while (depth != 0u)
			{
				const bool child_counter = ((child_counters >> depth) & 1u);

				const TreeNode& curr_node = m_nodes[node_idx[depth]];
				const uint32_t child_node_idx =
					curr_node.begin() +
					(child_counter ^ ((start_node >> curr_node.splitType()) & 1u));
				auto& child_node = m_nodes[child_node_idx];

				if (child_node.intersectsWith(ray))
				{
					if (child_node.isLeaf())
					{
						for (uint32_t i = child_node.begin(); i < child_node.end(); i++)
							m_container[i].ClosestIntersection(ray, traversal);
					}
					else
					{
						node_idx[++depth] = child_node_idx;
						child_counters &= ~(1u << depth);
						continue;
					}
				}

				child_counters |= 1u << depth;

				if (child_counter)
				{
					while ((child_counters >> --depth) & 1u);
					child_counters |= 1u << depth;
				}
			}
		}
		__device__ __inline__ ColorF AnyIntersection(const RangedRay& ray) const
		{
			if (m_count == 0u) return ColorF(1.0f);	// the tree is empty
			if (!m_nodes[0].intersectsWith(ray)) return ColorF(1.0f);	// ray misses root node

			ColorF shadow_mask(1.0f);

			// single node shortcut
			if (m_nodes[0].isLeaf())
			{
				// check all objects held by the node
				for (uint32_t i = m_nodes[0].begin(); i < m_nodes[0].end(); i++)
				{
					shadow_mask *= m_container[i].AnyIntersection(ray);
					if (shadow_mask.alpha < 0.0001f) return shadow_mask;
				}
				return shadow_mask;
			}

			uint8_t depth = 1;	// current depth
			uint32_t node_idx[32u];	// nodes in stack
			node_idx[depth] = 0u;
			uint32_t child_counters = 0u;

			while (depth != 0u)
			{
				const bool child_counter = ((child_counters >> depth) & 1u);

				const TreeNode& curr_node = m_nodes[node_idx[depth]];
				const uint32_t child_node_idx = curr_node.begin() + child_counter;
				auto& child_node = m_nodes[child_node_idx];

				if (child_node.intersectsWith(ray))
				{
					if (child_node.isLeaf())
					{
						for (uint32_t i = child_node.begin(); i < child_node.end(); i++)
						{
							shadow_mask *= m_container[i].AnyIntersection(ray);
							if (shadow_mask.alpha < 0.0001f) return shadow_mask;
						}
					}
					else
					{
						node_idx[++depth] = child_node_idx;
						child_counters &= ~(1u << depth);
						continue;
					}
				}

				child_counters |= 1u << depth;

				if (child_counter)
				{
					while ((child_counters >> --depth) & 1u);
					child_counters |= 1u << depth;
				}
			}

			return shadow_mask;
		}
	};
}

#endif // !CUDA_BVH_H