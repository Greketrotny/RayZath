#ifndef CUDA_BVH_TREE_NODE_CUH
#define CUDA_BVH_TREE_NODE_CUH

#include "cuda_render_parts.cuh"

namespace RayZath::Cuda
{
	struct TreeNode
	{
	private:
		static constexpr uint32_t type_mask = 0xC0000000;
		static constexpr uint32_t type_shift = 30;
		static constexpr uint32_t count_mask = 0x3FFFFFFF;

		BoundingBox m_bb;
		uint32_t m_begin = 0, m_count = 1;

	public:
		TreeNode() = default;
		__host__ TreeNode(
			const RayZath::Engine::BoundingBox& bb,
			const uint32_t split_type, const uint32_t begin, const uint32_t count)
			: m_bb(bb)
			, m_begin(begin)
			, m_count((split_type << type_shift & type_mask) | (count & count_mask))
		{}

	public:
		__host__ __device__ bool isLeaf() const
		{
			return count() != 0u;
		}
		__host__ __device__ uint32_t begin() const
		{
			return m_begin;
		}
		__host__ __device__ uint32_t count() const
		{
			return m_count & count_mask;
		}
		__host__ __device__ uint32_t end() const
		{
			return begin() + count();
		}
		__host__ __device__ uint32_t splitType() const
		{
			return (m_count & type_mask) >> type_shift;
		}

		__device__ bool intersectsWith(const RangedRay& ray) const
		{
			return m_bb.RayIntersection(ray);
		}
	};
}

#endif