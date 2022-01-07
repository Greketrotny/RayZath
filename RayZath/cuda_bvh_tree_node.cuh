#ifndef CUDA_BVH_TREE_NODE_CUH
#define CUDA_BVH_TREE_NODE_CUH

#include "cuda_render_parts.cuh"

namespace RayZath::Cuda
{
	struct TreeNode
	{
	private:
		BoundingBox m_bb;
		uint32_t m_begin, m_end; // if a tree node is a leaf, m_begin is an index of the first object 
								 // in the node and m_end is an index of the last object in the node.
								 // if a tree node is not a leaf, m_begin is an index of the first 
								 // child node and m_end is the index of the last child node.
		bool m_is_leaf;

	public:
		__host__ TreeNode()
			: m_begin(UINT32_MAX)
			, m_end(UINT32_MAX)
			, m_is_leaf(true)
		{}
		__host__ TreeNode(const RayZath::Engine::BoundingBox& bb, const bool is_leaf)
			: m_bb(bb)
			, m_begin(UINT32_MAX)
			, m_end(UINT32_MAX)
			, m_is_leaf(is_leaf)
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
		__device__ bool IntersectsWith(const RangedRay& ray) const
		{
			return m_bb.RayIntersection(ray);
		}
	};
}

#endif