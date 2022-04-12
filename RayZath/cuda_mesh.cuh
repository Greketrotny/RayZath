#ifndef CUDA_MESH_CUH
#define CUDA_MESH_CUH

#include "mesh.h"

#include "cuda_render_object.cuh"
#include "cuda_render_parts.cuh"
#include "cuda_bvh_tree_node.cuh"

namespace RayZath::Cuda
{
	struct MeshStructure
	{
	private:
		static HostPinnedMemory m_hpm_trs, m_hpm_nodes;
		static constexpr size_t sm_max_bvh_depth = 31u;

		Triangle* mp_triangles = nullptr;
		uint32_t m_triangle_capacity = 0u, m_triangle_count = 0u;

		TreeNode* mp_nodes = nullptr;
		uint32_t m_node_capacity = 0u, m_node_count = 0u;

	public:
		__host__ ~MeshStructure();

	public:
		__host__ void Reconstruct(
			const World& hCudaWorld,
			const RayZath::Engine::Handle<RayZath::Engine::MeshStructure>& hMeshStructure,
			cudaStream_t& mirror_stream);


	public:
		__device__ void ClosestIntersection(TriangleIntersection& intersection) const
		{
			if (m_node_count == 0u) return;	// the tree is empty
			if (!mp_nodes[0].intersectsWith(intersection.ray)) return;	// ray misses root node

			// single node shortcut
			if (mp_nodes[0].isLeaf())
			{
				for (uint32_t i = mp_nodes[0].begin(); i < mp_nodes[0].end(); i++)
					mp_triangles[i].ClosestIntersection(intersection);
				return;
			}

			// start node index (bit set means, this axis has flipped traversal order)
			const uint8_t start_node =
				(uint8_t(intersection.ray.direction.x < 0.0f) << 2u) |
				(uint8_t(intersection.ray.direction.y < 0.0f) << 1u) |
				(uint8_t(intersection.ray.direction.z < 0.0f));
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

				if (child_node.intersectsWith(intersection.ray))
				{
					if (child_node.isLeaf())
					{
						for (uint32_t i = child_node.begin(); i < child_node.end(); i++)
							mp_triangles[i].ClosestIntersection(intersection);
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
		__device__ ColorF AnyIntersection(
			TriangleIntersection& intersection,
			const Material* const* materials) const
		{
			if (m_node_count == 0u) return ColorF(1.0f);	// the tree is empty
			if (!mp_nodes[0].intersectsWith(intersection.ray)) return ColorF(1.0f);	// ray misses root node

			ColorF shadow_mask(1.0f);

			// single node shortcut
			if (mp_nodes[0].isLeaf())
			{
				for (uint32_t i = mp_nodes[0].begin(); i < mp_nodes[0].end(); i++)
				{
					if (mp_triangles[i].AnyIntersection(intersection))
					{
						const Texcrd texcrd = mp_triangles[i].TexcrdFromBarycenter(
							intersection.b1, intersection.b2);

						const Material* material = materials[mp_triangles[i].GetMaterialId()];
						shadow_mask *= material->GetOpacityColor(texcrd);
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

				if (child_node.intersectsWith(intersection.ray))
				{
					if (child_node.isLeaf())
					{
						for (uint32_t i = child_node.begin(); i < child_node.end(); i++)
						{
							if (mp_triangles[i].AnyIntersection(intersection))
							{
								const Texcrd texcrd = mp_triangles[i].TexcrdFromBarycenter(
									intersection.b1, intersection.b2);

								const Material* material = materials[mp_triangles[i].GetMaterialId()];
								shadow_mask *= material->GetOpacityColor(texcrd);
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


	class Mesh : public RenderObject
	{
	public:
		const MeshStructure* mesh_structure = nullptr;
		const Material* materials[RayZath::Engine::Mesh::GetMaterialCapacity()];

	public:
		__host__ Mesh();

	public:
		__host__ void Reconstruct(
			const World& hCudaWorld,
			const RayZath::Engine::Handle<RayZath::Engine::Mesh>& hMesh,
			cudaStream_t& mirror_stream);


	public:
		__device__ __inline__ void ClosestIntersection(RayIntersection& intersection) const
		{
			// [>] check ray intersection with bounding_box
			if (!bounding_box.RayIntersection(intersection.ray))
				return;

			// [>] transform object-space ray
			TriangleIntersection local_intersect;
			local_intersect.ray = intersection.ray;
			transformation.TransformRayG2L(local_intersect.ray);

			const float length_factor = local_intersect.ray.direction.Length();
			local_intersect.ray.near_far *= length_factor;
			local_intersect.ray.direction.Normalize();


			// Linear search
			/*for (uint32_t index = 0u;
				index < mesh_structure.GetTriangles().GetContainer().GetCount();
				++index)
			{
				const Triangle* triangle = &mesh_structure.GetTriangles().GetContainer()[index];
				triangle->ClosestIntersection(local_intersect);
			}*/
			// BVH search
			if (mesh_structure == nullptr) return;
			mesh_structure->ClosestIntersection(local_intersect);

			intersection.ray.color *= local_intersect.color;

			if (local_intersect.triangle)
			{
				// select material
				intersection.surface_material = materials[local_intersect.triangle->GetMaterialId()];

				// calculate texture coordinates
				intersection.texcrd =
					local_intersect.triangle->TexcrdFromBarycenter(
						local_intersect.b1, local_intersect.b2);

				intersection.ray.near_far = local_intersect.ray.near_far / length_factor;

				// calculate mapped normal
				vec3f mapped_normal;
				local_intersect.triangle->AverageNormal(local_intersect, mapped_normal);
				if (intersection.surface_material->GetNormalMap())
				{
					local_intersect.triangle->MapNormal(
						intersection.surface_material->GetNormalMap()->Fetch(intersection.texcrd),
						mapped_normal);
				}

				// calculate reverse normal factor (flip if looking at the other side of the triangle)
				const bool external = vec3f::DotProduct(
					local_intersect.triangle->GetNormal(),
					local_intersect.ray.direction) < 0.0f;
				const float external_factor = static_cast<float>(external) * 2.0f - 1.0f;

				// fill intersection normals
				intersection.surface_normal = local_intersect.triangle->GetNormal() * external_factor;
				intersection.mapped_normal = mapped_normal * external_factor;

				// set behind material
				intersection.behind_material = (external ? intersection.surface_material : nullptr);
				intersection.closest_object = this;
			}
		}
		__device__ __inline__ ColorF AnyIntersection(const RangedRay& ray) const
		{
			// [>] check ray intersection with bounding_box
			if (!bounding_box.RayIntersection(ray))
				return ColorF(1.0f);

			// [>] transpose objectSpaceRay
			RangedRay objectSpaceRay = ray;
			transformation.TransformRayG2L(objectSpaceRay);

			objectSpaceRay.near_far *= objectSpaceRay.direction.Length();
			objectSpaceRay.direction.Normalize();

			TriangleIntersection tri_intersection;
			tri_intersection.ray = objectSpaceRay;

			//float shadow = this->material.transmittance;
			if (mesh_structure == nullptr) return ColorF(1.0f);
			return mesh_structure->AnyIntersection(tri_intersection, this->materials);
		}
	};
}

#endif // !CUDA_MESH_CUH