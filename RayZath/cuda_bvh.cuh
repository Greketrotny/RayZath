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
		public:
			CudaTreeNode* m_child[8];
			bool m_is_leaf;
			uint32_t m_leaf_first_index, m_leaf_last_index;
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
			uint32_t m_nodes_capacity, m_nodes_count;

			CudaObject** m_ptrs;
			uint32_t m_ptrs_capacity, m_ptrs_count;


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

				uint32_t h_tree_size = hContainer.GetBVH().GetTreeSize();

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
			__host__ uint32_t CreateLeaf(uint32_t size)
			{
				if (m_ptrs_count + size > m_ptrs_capacity) return 0u;
				else
				{
					m_ptrs_count += size;
					return m_ptrs_count - size;
				}
			}
			__host__ void FillNode(
				CudaTreeNode* hCudaNode,
				const TreeNode<HostObject>& hNode,
				CudaTreeNode* hCudaTreeNodes,
				CudaObject** hCudaObjectPtrs,
				CudaObjectContainer<HostObject, CudaObject>& hCudaContainer)
			{
				if (hNode.IsLeaf())
				{
					uint32_t leaf_size = hNode.GetObjectCount();
					hCudaNode->m_leaf_first_index = CreateLeaf(leaf_size);
					hCudaNode->m_leaf_last_index = hCudaNode->m_leaf_first_index + leaf_size;
					for (uint32_t i = 0u; i < leaf_size; i++)
					{
						hCudaObjectPtrs[hCudaNode->m_leaf_first_index + i] =
							hCudaContainer.GetStorageAddress() + hNode.GetObject(i).GetResource()->GetId();
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
			__device__ __inline__ void ClosestIntersection(
				RayIntersection& intersection,
				const CudaRenderObject*& closest_object) const
			{
				if (m_nodes_count == 0u) return;	// the tree is empty
				if (!m_nodes[0].m_bb.RayIntersection(intersection.ray)) return;	// ray misses root node

				CudaTreeNode* node[8u];	// nodes in stack
				node[0] = &m_nodes[0];
				int8_t depth = 0;	// current depth
				// start node index (depends on ray direction)
				uint8_t start_node =
					(uint32_t(intersection.ray.direction.x > 0.0f) << 2u) |
					(uint32_t(intersection.ray.direction.y > 0.0f) << 1u) |
					(uint32_t(intersection.ray.direction.z > 0.0f));
				uint32_t child_counters = 0u;	// child counters mask (8 frames by 4 bits)


				while (depth >= 0 && depth < 7)
				{
					if (node[depth]->m_is_leaf)
					{
						// check all objects held by the node
						for (uint32_t i = node[depth]->m_leaf_first_index;
							i < node[depth]->m_leaf_last_index;
							i++)
						{
							if (m_ptrs[i]->ClosestIntersection(intersection))
								closest_object = m_ptrs[i];
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


					// get next child to check
					CudaTreeNode* child_node =
						node[depth]->m_child[((child_counters >> (4u * depth)) & 0b111u) ^ start_node];
					// increment checked child count
					child_counters += (1u << (4u * depth));

					if (child_node)
					{
						if (child_node->m_bb.RayIntersection(intersection.ray))
						{
							intersection.bvh_factor *= (1.0f -
								0.02f * float(((child_counters >> (4u * depth)) & 0b1111u)));

							// increment depth
							++depth;
							// set current node to its child
							node[depth] = child_node;
							// clear checked child counter
							child_counters &= (~(0b1111u << (4u * uint32_t(depth))));
						}
					}
				}
			}
			__device__ __inline__ ColorF AnyIntersection(
				const CudaRay& ray) const
			{
				if (m_nodes_count == 0u) return ColorF(1.0f);	// the tree is empty
				if (!m_nodes[0].m_bb.RayIntersection(ray)) return ColorF(1.0f);	// ray misses root node

				CudaTreeNode* node[8u];	// nodes in stack
				node[0] = &m_nodes[0];
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
					if (node[depth]->m_is_leaf)
					{
						// check all objects held by the node
						for (uint32_t i = node[depth]->m_leaf_first_index;
							i < node[depth]->m_leaf_last_index;
							i++)
						{
							shadow_mask *= m_ptrs[i]->AnyIntersection(ray);
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
							CudaTreeNode* child_node =
								node[depth]->m_child[((child_counters >> (4u * depth)) & 0b111u) ^ start_node];
							// increment checked child count
							child_counters += (1u << (4u * depth));

							if (child_node)
							{
								if (child_node->m_bb.RayIntersection(ray))
								{
									// increment depth
									++depth;
									// set current node to its child
									node[depth] = child_node;
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
				const CudaWorld& hCudaWorld,
				ObjectContainerWithBVH<HostObject>& hContainer,
				HostPinnedMemory& hpm,
				cudaStream_t& mirror_stream)
			{
				if (!hContainer.GetStateRegister().IsModified()) return;

				m_container.Reconstruct(hCudaWorld, hContainer, hpm, mirror_stream);
				m_bvh.Reconstruct(hContainer, m_container, hpm, mirror_stream);

				hContainer.GetStateRegister().MakeUnmodified();
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
}

#endif // !CUDA_BVH_H