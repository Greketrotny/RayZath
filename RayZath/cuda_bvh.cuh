#ifndef CUDA_BVH_H
#define CUDA_BVH_H

#include "bvh.h"
#include "cuda_object_container.cuh"
#include "cuda_render_parts.cuh"

#include "cuda_render_object.cuh"

namespace RayZath
{
	namespace CudaEngine
	{
		struct CudaTreeNode
		{
		private:
			uint32_t m_begin, m_end; // if a tree node is a leaf, m_begin is an index of the first object 
									 // in the node and m_end is an index of the last object in the node.
									 // if a tree node is not a leaf, m_begin is an index of the first 
									 // child node and m_end is the index of the last child node.
			CudaBoundingBox m_bb;
			bool m_is_leaf;

		public:
			__host__ CudaTreeNode()
				: m_is_leaf(true)
				, m_begin(UINT32_MAX)
				, m_end(UINT32_MAX)
			{}
			template <class HostObject>
			__host__ CudaTreeNode(const TreeNode<HostObject>& hNode)
				: m_is_leaf(hNode.IsLeaf())
				, m_bb(hNode.GetBoundingBox())
				, m_begin(UINT32_MAX)
				, m_end(UINT32_MAX)
			{}

		public:
			__host__ __device__ inline bool IsLeaf() const
			{
				return m_is_leaf;
			}
			__host__ __device__ inline uint32_t Begin() const
			{
				return m_begin;
			}
			__host__ __device__ inline uint32_t End() const
			{
				return m_end;
			}
			__host__ void SetRange(const uint32_t begin, const uint32_t end)
			{
				m_begin = begin;
				m_end = end;
			}

		public:
			__device__ bool IntersectsWith(const CudaRay& ray) const
			{
				return m_bb.RayIntersection(ray);
			}
		};

		template <class HostObject, class CudaObject>
		class CudaObjectContainerWithBVH
		{
		private:
			CudaObjectContainer<HostObject, CudaObject> m_container;

			CudaTreeNode* m_nodes;
			uint32_t m_capacity, m_count;


		public:
			__host__ CudaObjectContainerWithBVH()
				: m_nodes(nullptr)
				, m_capacity(0u)
				, m_count(0u)
			{}
			__host__ ~CudaObjectContainerWithBVH()
			{
				if (m_nodes) CudaErrorCheck(cudaFree(m_nodes));
				m_nodes = nullptr;
				m_capacity = 0u;
				m_count = 0u;
			}


		public:
			__host__ void Reconstruct(
				const CudaWorld& hCudaWorld,
				ObjectContainerWithBVH<HostObject>& hContainer,
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
				CudaTreeNode* hCudaTreeNodes =
					(CudaTreeNode*)malloc(m_capacity * sizeof(*hCudaTreeNodes));

				// construct BVH
				std::vector<uint32_t> reordered_ids;
				uint32_t object_count = 0u;
				new (&hCudaTreeNodes[(m_count = 0u)++]) CudaTreeNode(hContainer.GetBVH().GetRootNode());
				ConstructNode(
					hCudaTreeNodes[0],
					hCudaTreeNodes,
					hContainer.GetBVH().GetRootNode(),
					reordered_ids,
					object_count);

				// copy tree nodes constructed on host to device memory
				CudaErrorCheck(cudaMemcpy(
					m_nodes, hCudaTreeNodes,
					m_capacity * sizeof(CudaTreeNode),
					cudaMemcpyKind::cudaMemcpyHostToDevice));
				free(hCudaTreeNodes);

				// reconstruct linear container along reordered list of indices
				m_container.Reconstruct(hCudaWorld, hContainer, reordered_ids, hpm, mirror_stream);

				hContainer.GetStateRegister().MakeUnmodified();
			}
			__host__ void ConstructNode(
				CudaTreeNode& hCudaNode,
				CudaTreeNode* hCudaNodes,
				const TreeNode<HostObject>& hNode,
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
						const TreeNode<HostObject>* hChildNode = hNode.GetChild(hi);
						if (hChildNode)
						{
							CudaTreeNode& hChildCudaNode = hCudaNodes[hCudaNode.Begin() + i++];
							new (&hChildCudaNode) CudaTreeNode(*hChildNode);
							ConstructNode(hChildCudaNode, hCudaNodes, hNode, reordered_ids, object_count);
						}
					}
				}
			}


		public:
			__device__ __inline__ void ClosestIntersection(
				RayIntersection& intersection,
				const CudaRenderObject*& closest_object) const
			{
				if (m_count == 0u) return;	// the tree is empty
				if (!m_nodes[0].IntersectsWith(intersection.ray)) return;	// ray misses root node

				uint32_t node_idx[8u];	// nodes in stack
				node_idx[0] = 0u;
				int8_t depth = 0;	// current depth
				// start node index (depends on ray direction)
				uint8_t start_node =
					(uint32_t(intersection.ray.direction.x > 0.0f) << 2u) |
					(uint32_t(intersection.ray.direction.y > 0.0f) << 1u) |
					(uint32_t(intersection.ray.direction.z > 0.0f));
				uint32_t child_counters = 0u;	// child counters mask (8 frames by 4 bits)


				while (depth >= 0 && depth < 7)
				{
					if (m_nodes[node_idx[depth]].IsLeaf())
					{
						// check all objects held by the node
						for (uint32_t i = m_nodes[node_idx[depth]].Begin();
							i < m_nodes[node_idx[depth]].End();
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
					uint32_t child_node_idx =
						m_nodes[node_idx[depth]].Begin() +
						(((child_counters >> (4u * depth)) & 0b111u) ^ start_node);
					// increment checked child count
					child_counters += (1u << (4u * depth));

					if (child_node_idx != UINT32_MAX)
					{
						if (m_nodes[child_node_idx].IntersectsWith(intersection.ray))
						{
							intersection.bvh_factor *= (1.0f -
								0.02f * float(((child_counters >> (4u * depth)) & 0b1111u)));

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
				const CudaRay& ray) const
			{
				if (m_count == 0u) return ColorF(1.0f);	// the tree is empty
				if (!m_nodes[0].IntersectsWith(ray)) return ColorF(1.0f);	// ray misses root node

				uint32_t node_idx[8u];	// nodes in stack
				node_idx[0] = 0u;
				int8_t depth = 0;	// current depth
				// start node index (depends on ray direction)
				uint8_t start_node =
					(uint32_t(ray.direction.x > 0.0f) << 2u) |
					(uint32_t(ray.direction.y > 0.0f) << 1u) |
					(uint32_t(ray.direction.z > 0.0f));
				uint32_t child_counters = 0u;	// child counters mask (8 frames by 4 bits)

				ColorF shadow_mask(1.0f);

				while (depth >= 0 && depth < 7u)
				{
					if (m_nodes[node_idx[depth]].IsLeaf())
					{
						// check all objects held by the node
						for (uint32_t i = m_nodes[node_idx[depth]].Begin();
							i < m_nodes[node_idx[depth]].End();
							i++)
						{
							shadow_mask *= m_container[i].AnyIntersection(ray);
							if (shadow_mask.alpha < 0.0001f) return shadow_mask;
						}
						--depth;
					}
					else
					{
						// check checked child count
						if (((child_counters >> (4u * depth)) & 0b1111u) >= 8u)
						{	// all children checked - decrement depth
							--depth;
						}
						else
						{
							// get next child to check
							uint32_t child_node_idx =
								m_nodes[node_idx[depth]].Begin() +
								(((child_counters >> (4u * depth)) & 0b111u) ^ start_node);
							// increment checked child count
							child_counters += (1u << (4u * depth));

							if (child_node_idx != UINT32_MAX)
							{
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
						}
					}
				}

				return shadow_mask;
			}
		};
	}
}

#endif // !CUDA_BVH_H