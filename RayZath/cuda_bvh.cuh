#ifndef CUDA_BVH_H
#define CUDA_BVH_H

#include "bvh.h"
#include "cuda_object_container.cuh"
#include "cuda_render_parts.cuh"

#include "cuda_render_object.cuh"
#include "cuda_bvh_tree_node.cuh"

namespace RayZath::Cuda
{
		template <class HostObject, class CudaObject>
		class ObjectContainerWithBVH
		{
		private:
			ObjectContainer<HostObject, CudaObject> m_container;

			TreeNode* m_nodes;
			uint32_t m_capacity, m_count;


		public:
			__host__ ObjectContainerWithBVH()
				: m_nodes(nullptr)
				, m_capacity(0u)
				, m_count(0u)
			{}
			__host__ ~ObjectContainerWithBVH()
			{
				if (m_nodes) CudaErrorCheck(cudaFree(m_nodes));
				m_nodes = nullptr;
				m_capacity = 0u;
				m_count = 0u;
			}


		public:
			__host__ void Reconstruct(
				const World& hCudaWorld,
				RayZath::Engine::ObjectContainerWithBVH<HostObject>& hContainer,
				HostPinnedMemory& hpm,
				cudaStream_t& mirror_stream)
			{
				if (!hContainer.GetStateRegister().IsModified()) return;

				const uint32_t tree_size = hContainer.GetBVH().GetTreeSize();
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

				// allocate host memory to construct nodes
				TreeNode* hCudaTreeNodes =
					(TreeNode*)malloc(m_capacity * sizeof(*hCudaTreeNodes));

				// construct BVH
				std::vector<uint32_t> reordered_ids;
				uint32_t object_count = 0u;
				const auto& hRootNode = hContainer.GetBVH().GetRootNode();
				new (&hCudaTreeNodes[(m_count = 0u)++]) TreeNode(hRootNode.GetBoundingBox(), hRootNode.IsLeaf());
				ConstructNode(
					hCudaTreeNodes[0],
					hCudaTreeNodes,
					hRootNode,
					reordered_ids,
					object_count);

				// copy tree nodes constructed on host to device memory
				CudaErrorCheck(cudaMemcpy(
					m_nodes, hCudaTreeNodes,
					m_capacity * sizeof(TreeNode),
					cudaMemcpyKind::cudaMemcpyHostToDevice));
				free(hCudaTreeNodes);

				// reconstruct linear container along reordered list of indices
				m_container.Reconstruct(hCudaWorld, hContainer, reordered_ids, hpm, mirror_stream);

				hContainer.GetStateRegister().MakeUnmodified();
			}
			__host__ void ConstructNode(
				TreeNode& hCudaNode,
				TreeNode* hCudaNodes,
				const RayZath::Engine::TreeNode<HostObject>& hNode,
				std::vector<uint32_t>& reordered_ids,
				uint32_t& object_count)
			{
				if (hNode.IsLeaf())
				{
					const uint32_t leaf_size = hNode.GetObjectCount();
					hCudaNode.SetRange(object_count, object_count + leaf_size);
					object_count += leaf_size;
					for (uint32_t i = 0u; i < leaf_size; i++)
					{
						reordered_ids.push_back(hNode.GetObject(i).GetAccessor()->GetIdx());
					}
				}
				else
				{
					const uint32_t child_count = hNode.GetChildCount();
					hCudaNode.SetRange(m_count, m_count + child_count);
					m_count += child_count;
					for (uint32_t hi = 0u, i = 0u; hi < 8u; hi++)
					{
						const auto* hChildNode = hNode.GetChild(hi);
						if (hChildNode)
						{
							TreeNode& hChildCudaNode = hCudaNodes[hCudaNode.Begin() + i++];
							new (&hChildCudaNode) TreeNode(hChildNode->GetBoundingBox(), hChildNode->IsLeaf());
							ConstructNode(hChildCudaNode, hCudaNodes, *hChildNode, reordered_ids, object_count);
						}
					}
				}
			}


		public:
			__device__ __inline__ void ClosestIntersection(
				RayIntersection& intersection,
				const RenderObject*& closest_object) const
			{
				if (m_count == 0u) return;	// the tree is empty
				if (!m_nodes[0].IntersectsWith(intersection.ray)) return;	// ray misses root node

				int8_t depth = 0;	// current depth
				uint32_t node_idx[8u];	// nodes in stack
				node_idx[0] = 0u;
				// start node index (depends on ray direction)
				const uint8_t start_node =
					(uint8_t(intersection.ray.direction.x > 0.0f) << 2u) |
					(uint8_t(intersection.ray.direction.y > 0.0f) << 1u) |
					(uint8_t(intersection.ray.direction.z > 0.0f));
				uint32_t child_counters = 0u;	// child counters mask (8 frames by 4 bits)


				while (depth >= 0 && depth < 7)
				{
					const TreeNode& curr_node = m_nodes[node_idx[depth]];
					if (curr_node.IsLeaf())
					{
						// check all objects held by the node
						for (uint32_t i = curr_node.Begin();
							i < curr_node.End();
							i++)
						{
							if (m_container[i].ClosestIntersection(intersection))
								closest_object = &m_container[i];
						}
						--depth;
						continue;
					}

					// check checked child count
					if (((child_counters >> (4u * depth)) & 0b1111u) >= 8u)
					{	// all children checked - decrement depth

						--depth;
						continue;
					}


					// get next child node idx to check
					const uint32_t child_node_idx =
						curr_node.Begin() +
						(((child_counters >> (4u * depth)) & 0b111u) ^ start_node);
					// increment checked child count
					child_counters += (1u << (4u * depth));

					if (child_node_idx < curr_node.End())
					{
						if (m_nodes[child_node_idx].IntersectsWith(intersection.ray))
						{
							// increment depth
							++depth;
							// set current node to its child
							node_idx[depth] = child_node_idx;
							// clear checked child counter
							child_counters &= (~(0b1111u << (4u * uint32_t(depth))));
						}
					}
				}
			}
			__device__ __inline__ ColorF AnyIntersection(
				const RangedRay& ray) const
			{
				if (m_count == 0u) return ColorF(1.0f);	// the tree is empty
				if (!m_nodes[0].IntersectsWith(ray)) return ColorF(1.0f);	// ray misses root node

				int8_t depth = 0;	// current depth
				uint32_t node_idx[8u];	// nodes in stack
				node_idx[0] = 0u;
				uint32_t child_counters = 0u;	// child counters mask (8 frames by 4 bits)

				ColorF shadow_mask(1.0f);

				while (depth >= 0 && depth < 7u)
				{
					const TreeNode& curr_node = m_nodes[node_idx[depth]];
					if (curr_node.IsLeaf())
					{
						// check all objects held by the node
						for (uint32_t i = curr_node.Begin();
							i < curr_node.End();
							i++)
						{
							shadow_mask *= m_container[i].AnyIntersection(ray);
							if (shadow_mask.alpha < 0.0001f) return shadow_mask;
						}
						--depth;
						continue;
					}

					// get next child node idx to check
					const uint32_t child_node_idx =
						curr_node.Begin() +
						((child_counters >> (4u * depth)) & 0b1111u);
					if (child_node_idx >= curr_node.End())
					{
						--depth;
						continue;
					}

					// increment checked child count
					child_counters += (1u << (4u * depth));

					if (m_nodes[child_node_idx].IntersectsWith(ray))
					{
						// increment depth
						++depth;
						// set current node to its child
						node_idx[depth] = child_node_idx;
						// clear checked child counter
						child_counters &= (~(0b1111u << (4u * uint32_t(depth))));
					}
				}

				return shadow_mask;
			}
		};
	
}

#endif // !CUDA_BVH_H