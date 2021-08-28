#include "mesh_component.h"

#include "mesh_structure.h"

namespace RayZath
{
	// ~~~~~~~~ [STRUCT] Triangle ~~~~~~~~
	Triangle::Triangle(
		const std::array<uint32_t, 3u>& vs,
		const std::array<uint32_t, 3u>& ts,
		const std::array<uint32_t, 3u>& ns,
		const uint32_t& mat_id)
		: vertices(vs)
		, texcrds(ts)
		, normals(ns)
		, material_id(mat_id)
	{}

	void Triangle::CalculateNormal(const MeshStructure& mesh_structure)
	{
		const Math::vec3f& v1 = mesh_structure.GetVertices()[vertices[0]];
		const Math::vec3f& v2 = mesh_structure.GetVertices()[vertices[1]];
		const Math::vec3f& v3 = mesh_structure.GetVertices()[vertices[2]];
		normal = Math::vec3f::CrossProduct(v2 - v3, v2 - v1);
		normal.Normalize();
	}
	BoundingBox Triangle::GetBoundingBox(const MeshStructure& mesh_structure) const
	{
		const Math::vec3f& v1 = mesh_structure.GetVertices()[vertices[0]];
		const Math::vec3f& v2 = mesh_structure.GetVertices()[vertices[1]];
		const Math::vec3f& v3 = mesh_structure.GetVertices()[vertices[2]];
		return BoundingBox(v1, v2, v3);
	}
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
}