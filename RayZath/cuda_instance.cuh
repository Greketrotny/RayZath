#ifndef CUDA_INSTANCE_CUH
#define CUDA_INSTANCE_CUH

#include "cuda_render_parts.cuh"
#include "cuda_bvh_tree_node.cuh"
#include "cuda_material.cuh"
#include "cuda_engine_parts.cuh"

namespace RayZath::Engine
{
	class World;
	class Mesh;
	class Instance;

	template <typename T>
	struct Handle;
}

namespace RayZath::Cuda
{
	struct Mesh
	{
	private:
		static HostPinnedMemory m_hpm_trs, m_hpm_nodes;
		static constexpr size_t sm_max_bvh_depth = 31u;

		Triangle* mp_triangles = nullptr;
		uint32_t m_triangle_capacity = 0u, m_triangle_count = 0u;

		TreeNode* mp_nodes = nullptr;
		uint32_t m_node_capacity = 0u, m_node_count = 0u;

	public:
		__host__ ~Mesh();

	public:
		__host__ void reconstruct(
			const World& hCudaWorld,
			const RayZath::Engine::Handle<RayZath::Engine::Mesh>& hMesh,
			cudaStream_t& mirror_stream);


	public:
		__device__ void closestIntersection(RangedRay& ray, TraversalResult& traversal) const
		{
			if (m_node_count == 0u) return;	// the tree is empty
			if (!mp_nodes[0].intersectsWith(ray)) return;	// ray misses root node

			// single node shortcut
			if (mp_nodes[0].isLeaf())
			{
				for (uint32_t i = mp_nodes[0].begin(); i < mp_nodes[0].end(); i++)
					mp_triangles[i].closestIntersection(ray, traversal);
				return;
			}

			// start node index (bit set means, this axis has flipped traversal order)
			const uint8_t start_node =
				(uint8_t(ray.direction.x < 0.0f) << 2u) |
				(uint8_t(ray.direction.y < 0.0f) << 1u) |
				(uint8_t(ray.direction.z < 0.0f));
			int8_t depth = 1;	// current depth
			uint32_t node_idx[32u];	// nodes in stack
			node_idx[depth] = 0u;
			uint32_t child_counters = 0u;

			while (depth != 0u)
			{
				const bool child_counter = ((child_counters >> depth) & 1u);

				const TreeNode& curr_node = mp_nodes[node_idx[depth]];
				const uint32_t child_node_idx =
					curr_node.begin() +
					(child_counter ^ ((start_node >> curr_node.splitType()) & 1u));
				auto& child_node = mp_nodes[child_node_idx];

				if (child_node.intersectsWith(ray))
				{
					if (child_node.isLeaf())
					{
						for (uint32_t i = child_node.begin(); i < child_node.end(); i++)
							mp_triangles[i].closestIntersection(ray, traversal);
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
		__device__ ColorF anyIntersection(RangedRay& ray, const Material* const* materials) const
		{
			if (m_node_count == 0u) return ColorF(1.0f);	// the tree is empty
			if (!mp_nodes[0].intersectsWith(ray)) return ColorF(1.0f);	// ray misses root node

			ColorF shadow_mask(1.0f);
			vec2f barycenter;

			// single node shortcut
			if (mp_nodes[0].isLeaf())
			{
				for (uint32_t i = mp_nodes[0].begin(); i < mp_nodes[0].end(); i++)
				{
					if (mp_triangles[i].anyIntersection(ray, barycenter))
					{
						const Texcrd texcrd = mp_triangles[i].texcrdFromBarycenter(barycenter);

						const Material* material = materials[mp_triangles[i].materialId()];
						shadow_mask *= material->opacityColor(texcrd);
						if (shadow_mask.alpha < 1.0e-4f) return shadow_mask;
					}
				}
				return shadow_mask;
			}

			int8_t depth = 1;	// current depth
			uint32_t node_idx[32u];	// nodes in stack
			node_idx[depth] = 0u;
			uint32_t child_counters = 0u;

			while (depth != 0u)
			{
				const bool child_counter = ((child_counters >> depth) & 1u);

				const TreeNode& curr_node = mp_nodes[node_idx[depth]];
				const uint32_t child_node_idx = curr_node.begin() + child_counter;
				auto& child_node = mp_nodes[child_node_idx];

				if (child_node.intersectsWith(ray))
				{
					if (child_node.isLeaf())
					{
						for (uint32_t i = child_node.begin(); i < child_node.end(); i++)
						{
							if (mp_triangles[i].anyIntersection(ray, barycenter))
							{
								const Texcrd texcrd = mp_triangles[i].texcrdFromBarycenter(barycenter);

								const Material* material = materials[mp_triangles[i].materialId()];
								shadow_mask *= material->opacityColor(texcrd);
								if (shadow_mask.alpha < 1.0e-4f) return shadow_mask;
							}
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

	class Instance
	{
	public:
		Transformation transformation;
		BoundingBox bounding_box;

		const Mesh* mesh = nullptr;
		const Material* materials[0x100];

		uint32_t m_instance_idx;

		__host__ Instance();

		__host__ void reconstruct(
			const World& hCudaWorld,
			const RayZath::Engine::Handle<RayZath::Engine::Instance>& hInstance,
			cudaStream_t& mirror_stream);


		__device__ __inline__ void closestIntersection(RangedRay& ray, TraversalResult& traversal) const
		{
			// [>] check ray intersection with bounding_box
			if (!bounding_box.rayIntersection(ray))
				return;

			// [>] transform object-space ray
			RangedRay local_ray = ray;
			transformation.transformG2L(local_ray);

			const float length_factor = local_ray.direction.Length();
			local_ray.near_far *= length_factor;
			local_ray.direction.Normalize();

			if (mesh == nullptr) return;
			const auto* const closest_triangle = traversal.closest_triangle;
			traversal.closest_triangle = nullptr;
			mesh->closestIntersection(local_ray, traversal);

			if (traversal.closest_triangle)
			{
				traversal.closest_instance = this;
				ray.near_far = local_ray.near_far / length_factor;
			}
			else
			{
				traversal.closest_triangle = closest_triangle;
			}
		}
		__device__ __inline__ ColorF anyIntersection(const RangedRay& ray) const
		{
			// [>] check ray intersection with bounding_box
			if (!bounding_box.rayIntersection(ray))
				return ColorF(1.0f);

			// [>] transpose objectSpaceRay
			RangedRay local_ray = ray;
			transformation.transformG2L(local_ray);
			local_ray.near_far *= local_ray.direction.Length();
			local_ray.direction.Normalize();

			if (mesh == nullptr) return ColorF(1.0f);
			return mesh->anyIntersection(local_ray, this->materials);
		}

		__device__ void analyzeIntersection(TraversalResult& traversal, SurfaceProperties& surface) const
		{
			// select materials
			surface.surface_material = materials[traversal.closest_triangle->materialId()];
			if (traversal.external) surface.behind_material = surface.surface_material;
			
			// calculate texture coordinates
			surface.texcrd = traversal.closest_triangle->texcrdFromBarycenter(traversal.barycenter);

			const float external_factor = static_cast<float>(traversal.external) * 2.0f - 1.0f;

			// calculate mapped normal
			surface.mapped_normal = traversal.closest_triangle->averageNormal(traversal.barycenter);
			if (surface.surface_material->normalMap())
			{
				traversal.closest_triangle->mapNormal(
					surface.surface_material->normalMap()->fetch(surface.texcrd),
					surface.mapped_normal,
					transformation.scale);
				transformation.transformL2GNoScale(surface.mapped_normal);
			}
			else
			{
				transformation.transformL2G(surface.mapped_normal);
			}
			surface.mapped_normal.Normalize();
			surface.mapped_normal *= external_factor;

			// calculate surface normal (triangle based)
			surface.normal = traversal.closest_triangle->getNormal() * external_factor;
			transformation.transformL2G(surface.normal);
			surface.normal.Normalize();
		}
	};
}

#endif // !CUDA_MESH_CUH