#include "mesh_component.hpp"

#include "cpu_render_utils.hpp"
#include "mesh.hpp"

namespace RayZath::Engine
{
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

	void Triangle::calculateNormal(const Mesh& mesh)
	{
		const Math::vec3f& v1 = mesh.vertices()[vertices[0]];
		const Math::vec3f& v2 = mesh.vertices()[vertices[1]];
		const Math::vec3f& v3 = mesh.vertices()[vertices[2]];
		normal = Math::vec3f::CrossProduct(v2 - v3, v2 - v1);
		normal.Normalize();
	}
	BoundingBox Triangle::boundingBox(const Mesh& mesh) const
	{
		const Math::vec3f& v1 = mesh.vertices()[vertices[0]];
		const Math::vec3f& v2 = mesh.vertices()[vertices[1]];
		const Math::vec3f& v3 = mesh.vertices()[vertices[2]];
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

	void Triangle::closestIntersection(CPU::RangedRay& ray, CPU::TraversalResult& traversal, const Mesh& mesh) const
	{		
		const auto v1{mesh.vertices()[vertices[0]]};
		const auto v2{mesh.vertices()[vertices[1]]};
		const auto v3{mesh.vertices()[vertices[2]]};
		const auto edge1 = v2 - v1;
		const auto edge2 = v3 - v1;
		const auto pvec = Math::vec3f32::CrossProduct(ray.direction, edge2);

		float det = Math::vec3f32::DotProduct(edge1, pvec);
		det += static_cast<float>(uint8_t(det > -1.0e-7f) & uint8_t(det < 1.0e-7f)) * 1.0e-7f;
		const float inv_det = 1.0f / det;

		const Math::vec3f32 tvec = ray.origin - v1;
		const float b1 = Math::vec3f32::DotProduct(tvec, pvec) * inv_det;
		if (b1 < 0.0f || b1 > 1.0f)
			return;

		const Math::vec3f32 qvec = Math::vec3f32::CrossProduct(tvec, edge1);
		const float b2 = Math::vec3f32::DotProduct(ray.direction, qvec) * inv_det;
		if (b2 < 0.0f || b1 + b2 > 1.0f)
			return;

		const float t = Math::vec3f32::DotProduct(edge2, qvec) * inv_det;
		if (t <= ray.near_far.x || t >= ray.near_far.y)
			return;

		ray.near_far.y = t;
		traversal.closest_triangle = this;
		traversal.external = det > 0.0f;
		traversal.barycenter = Math::vec2f32(b1, b2);
	}
	bool Triangle::anyIntersection(const CPU::RangedRay& ray, Math::vec2f32& barycenter, const Mesh& mesh) const
	{
		const auto v1{mesh.vertices()[vertices[0]]};
		const auto v2{mesh.vertices()[vertices[1]]};
		const auto v3{mesh.vertices()[vertices[2]]};
		const auto edge1 = v2 - v1;
		const auto edge2 = v3 - v1;
		const auto pvec = Math::vec3f32::CrossProduct(ray.direction, edge2);

		float det = Math::vec3f32::DotProduct(edge1, pvec);
		det += static_cast<float>(uint8_t(det > -1.0e-7f) & uint8_t(det < 1.0e-7f)) * 1.0e-7f;
		const float inv_det = 1.0f / det;

		const Math::vec3f32 tvec = ray.origin - v1;
		const float b1 = Math::vec3f32::DotProduct(tvec, pvec) * inv_det;
		if (b1 < 0.0f || b1 > 1.0f)
			return false;

		const Math::vec3f32 qvec = Math::vec3f32::CrossProduct(tvec, edge1);
		const float b2 = Math::vec3f32::DotProduct(ray.direction, qvec) * inv_det;
		if (b2 < 0.0f || b1 + b2 > 1.0f)
			return false;

		const float t = Math::vec3f32::DotProduct(edge2, qvec) * inv_det;
		if (t <= ray.near_far.x || t >= ray.near_far.y)
			return false;

		barycenter = Math::vec2f32(b1, b2);

		return true;
	}
	Texcrd Triangle::texcrdFromBarycenter(const Math::vec2f32 barycenter, const Mesh& mesh) const
	{
		const auto t1{mesh.texcrds()[texcrds[0]]};
		const auto t2{mesh.texcrds()[texcrds[1]]};
		const auto t3{mesh.texcrds()[texcrds[2]]};
		const float b3 = 1.0f - barycenter.x - barycenter.y;
		const float u = t1.x * b3 + t2.x * barycenter.x + t3.x * barycenter.y;
		const float v = t1.y * b3 + t2.y * barycenter.x + t3.y * barycenter.y;
		return Texcrd(u, v);
	}
	Math::vec3f32 Triangle::averageNormal(const Math::vec2f32 barycenter, const Mesh& mesh) const
	{
		const auto n1{mesh.normals()[normals[0]]};
		const auto n2{mesh.normals()[normals[1]]};
		const auto n3{mesh.normals()[normals[2]]};
		return  (n1 * (1.0f - barycenter.x - barycenter.y) + n2 * barycenter.x + n3 * barycenter.y).Normalized();
	}
	void Triangle::mapNormal(
		const Graphics::ColorF& map_color, 
		Math::vec3f32& mapped_normal, 
		const Mesh& mesh,
		const Math::vec3f32& scale) const
	{
		const auto v1{mesh.vertices()[vertices[0]]};
		const auto v2{mesh.vertices()[vertices[1]]};
		const auto v3{mesh.vertices()[vertices[2]]};

		const auto t1{mesh.texcrds()[texcrds[0]]};
		const auto t2{mesh.texcrds()[texcrds[1]]};
		const auto t3{mesh.texcrds()[texcrds[2]]};

		const Math::vec3f32 edge1 = (v2 - v1) * scale;
		const Math::vec3f32 edge2 = (v3 - v1) * scale;
		const Math::vec2f32 dUV1 = t2 - t1;
		const Math::vec2f32 dUV2 = t3 - t1;
		mapped_normal /= scale;

		// tangent and bitangent
		const float f = 1.0f / (dUV1.x * dUV2.y - dUV2.x * dUV1.y);
		Math::vec3f32 tangent = ((edge1 * dUV2.y - edge2 * dUV1.y) * f).Normalized();
		// tangent re-orthogonalization
		tangent = (tangent - mapped_normal * Math::vec3f32::DotProduct(tangent, mapped_normal)).Normalized();
		// bitangent is simply cross product of normal and tangent
		Math::vec3f32 bitangent = Math::vec3f32::CrossProduct(tangent, mapped_normal);

		// map normal transformation to [-1.0f, 1.0f] range
		const Math::vec3f32 map_normal = 
			Math::vec3f32(map_color.red, map_color.green, map_color.blue) * 2.0f - 
			Math::vec3f32(1.0f);

		// calculate normal
		mapped_normal = mapped_normal * map_normal.z + tangent * map_normal.x + bitangent * map_normal.y;
	}
}