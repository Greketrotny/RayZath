#ifndef CUDA_MESH_CUH
#define CUDA_MESH_CUH

#include "mesh.h"

#include "cuda_render_object.cuh"
#include "cuda_render_parts.cuh"
#include "cuda_bvh_tree_node.cuh"

namespace RayZath
{
	namespace CudaEngine
	{
		struct CudaMeshStructure
		{
		private:
			static HostPinnedMemory m_hpm_trs, m_hpm_nodes;
			static constexpr size_t sm_max_bvh_depth = 16u;
			static constexpr size_t sm_max_child_count = 8u;

			CudaTriangle* mp_triangles;
			uint32_t m_triangle_capacity, m_triangle_count;

			CudaTreeNode* mp_nodes;
			uint32_t m_node_capacity, m_node_count;

		public:
			__host__ CudaMeshStructure();
			__host__ ~CudaMeshStructure();


		public:
			__host__ void Reconstruct(
				const CudaWorld& hCudaWorld,
				const Handle<MeshStructure>& hMeshStructure,
				cudaStream_t& mirror_stream);


		public:
			__device__ void ClosestIntersection(TriangleIntersection& intersection) const
			{
				if (m_node_count == 0u) return;	// the tree is empty
				if (!mp_nodes[0].IntersectsWith(intersection.ray)) return;	// ray misses root node

				int8_t depth = 0;	// current depth
				uint32_t node_idx[16];	// nodes in stack
				node_idx[0] = 0u; // node at depth 0 -> root node
				// start node index (depends on ray direction)
				const uint8_t start_node =
					(uint8_t(intersection.ray.direction.x > 0.0f) << 2u) |
					(uint8_t(intersection.ray.direction.y > 0.0f) << 1u) |
					(uint8_t(intersection.ray.direction.z > 0.0f));
				uint64_t child_counters = 0u;	// child counters mask (16 frames by 4 bits)

				while (depth >= 0 && depth < 15)
				{
					const CudaTreeNode& curr_node = mp_nodes[node_idx[depth]];
					if (curr_node.IsLeaf())
					{
						// check all objects held by the node
						for (uint32_t i = curr_node.Begin();
							i < curr_node.End();
							i++)
						{
							mp_triangles[i].ClosestIntersection(intersection);
						}
						--depth;
						continue;
					}

					// check checked child count
					if (((child_counters >> (4ull * depth)) & 0b1111ull) >= 8ull)
					{	// all children checked - decrement depth

						--depth;
						continue;
					}

					// get next child node idx to check
					const uint32_t child_node_idx =
						curr_node.Begin() +
						(((child_counters >> (4ull * depth)) & 0b111ull) ^ start_node);
					// increment checked children count
					child_counters += (1ull << (4ull * depth));

					if (child_node_idx < curr_node.End())
					{
						if (mp_nodes[child_node_idx].IntersectsWith(intersection.ray))
						{
							intersection.bvh_factor *= (1.0f -
								0.01f * float(((child_counters >> (4ull * depth)) & 0b1111ull)));

							// increment depth
							++depth;
							// set current node to its child
							node_idx[depth] = child_node_idx;
							// clear checked children counter
							child_counters &= (~(0b1111ull << (4ull * depth)));
						}
					}
				}
			}
			__device__ ColorF AnyIntersection(
				TriangleIntersection& intersection,
				const CudaMaterial* const* materials) const
			{
				if (m_node_count == 0u) return ColorF(1.0f);	// the tree is empty
				if (!mp_nodes[0].IntersectsWith(intersection.ray)) return ColorF(1.0f);	// ray misses root node

				int8_t depth = 0;	// current depth
				uint32_t node_idx[16u]; // nodes in stack
				node_idx[0] = 0u;  // node at depth 0 -> root node
				uint64_t child_counters = 0u;	// child counters mask (16 frames by 4 bits)

				ColorF shadow_mask(1.0f);

				while (depth >= 0 && depth < 15)
				{
					const CudaTreeNode& curr_node = mp_nodes[node_idx[depth]];
					if (curr_node.IsLeaf())
					{
						// check all objects held by the node
						for (uint32_t i = curr_node.Begin();
							i < curr_node.End();
							i++)
						{
							if (mp_triangles[i].AnyIntersection(intersection))
							{
								const CudaTexcrd texcrd = mp_triangles[i].TexcrdFromBarycenter(
									intersection.b1, intersection.b2);

								const CudaMaterial* material = materials[mp_triangles[i].material_id];
								shadow_mask *= material->GetOpacityColor(texcrd);
								if (shadow_mask.alpha < 1.0e-4f) return shadow_mask;
							}
						}
						--depth;
						continue;
					}

					// get next child node idx to check
					const uint32_t child_node_idx =
						curr_node.Begin() +
						((child_counters >> (4ull * depth)) & 0b1111ull);
					if (child_node_idx >= curr_node.End())
					{
						--depth;
						continue;
					}

					// increment checked children count
					child_counters += (1ull << (4ull * depth));

					if (mp_nodes[child_node_idx].IntersectsWith(intersection.ray))
					{
						// increment depth
						++depth;
						// set current node to its child
						node_idx[depth] = child_node_idx;
						// clear checked children counter
						child_counters &= (~(0b1111ull << (4ull * depth)));
					}
				}

				return shadow_mask;
			}
		};


		class CudaMesh : public CudaRenderObject
		{
		public:
			const CudaMeshStructure* mesh_structure;
			const CudaMaterial* materials[Mesh::GetMaterialCapacity()];


		public:
			__host__ CudaMesh();


		public:
			__host__ void Reconstruct(
				const CudaWorld& hCudaWorld,
				const Handle<Mesh>& hMesh,
				cudaStream_t& mirror_stream);


			// device rendering functions
		public:
			__device__ __inline__ bool ClosestIntersection(RayIntersection& intersection) const
			{
				// [>] check ray intersection with bounding_box
				if (!bounding_box.RayIntersection(intersection.ray))
					return false;


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
					const CudaTriangle* triangle = &mesh_structure.GetTriangles().GetContainer()[index];
					triangle->ClosestIntersection(local_intersect);
				}*/
				// BVH search
				if (mesh_structure == nullptr) return false;
				mesh_structure->ClosestIntersection(local_intersect);


				// ~~~~ BVH debug
				intersection.bvh_factor *= local_intersect.bvh_factor;
				// ~~~~ BVH debug


				if (local_intersect.triangle)
				{
					// select material
					intersection.surface_material = materials[local_intersect.triangle->material_id];

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
					const bool reverse = vec3f::DotProduct(
						local_intersect.triangle->normal,
						local_intersect.ray.direction) < 0.0f;
					const float reverse_factor = static_cast<float>(reverse) * 2.0f - 1.0f;

					// fill intersection normals
					intersection.surface_normal = local_intersect.triangle->normal * reverse_factor;
					intersection.mapped_normal = mapped_normal * reverse_factor;


					// set material
					if (!reverse)
					{	// intersection from inside

						intersection.behind_material = nullptr;
					}
					else
					{	// intersection from outside

						intersection.behind_material =
							intersection.surface_material;
					}

					return true;
				}

				return false;
			}
			__device__ __inline__ ColorF AnyIntersection(const CudaRay& ray) const
			{
				// [>] check ray intersection with bounding_box
				if (!bounding_box.RayIntersection(ray))
					return ColorF(1.0f);

				// [>] transpose objectSpaceRay
				CudaRay objectSpaceRay = ray;
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
}

#endif // !CUDA_MESH_CUH