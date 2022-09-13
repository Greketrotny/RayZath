#include "mesh_component.hpp"

#include "mesh_structure.hpp"

namespace RayZath::Engine
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

	void Triangle::calculateNormal(const MeshStructure& mesh_structure)
	{
		const Math::vec3f& v1 = mesh_structure.vertices()[vertices[0]];
		const Math::vec3f& v2 = mesh_structure.vertices()[vertices[1]];
		const Math::vec3f& v3 = mesh_structure.vertices()[vertices[2]];
		normal = Math::vec3f::CrossProduct(v2 - v3, v2 - v1);
		normal.Normalize();
	}
	BoundingBox Triangle::boundingBox(const MeshStructure& mesh_structure) const
	{
		const Math::vec3f& v1 = mesh_structure.vertices()[vertices[0]];
		const Math::vec3f& v2 = mesh_structure.vertices()[vertices[1]];
		const Math::vec3f& v3 = mesh_structure.vertices()[vertices[2]];
		return BoundingBox(v1, v2, v3);
	}


	bool Triangle::areVertsValid() const
	{
		constexpr uint32_t npos = std::numeric_limits<uint32_t>::max();
		return vertices[0] != npos && vertices[1] != npos && vertices[2] != npos;
	}
	bool Triangle::areTexcrdsValid() const
	{
		constexpr uint32_t npos = std::numeric_limits<uint32_t>::max();
		return texcrds[0] != npos && texcrds[1] != npos && texcrds[2] != npos;
	}
	bool Triangle::areNormalsValid() const
	{
		constexpr uint32_t npos = std::numeric_limits<uint32_t>::max();
		return normals[0] != npos && normals[1] != npos && normals[2] != npos;
	}
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
}