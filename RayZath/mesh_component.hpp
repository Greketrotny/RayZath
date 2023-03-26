#ifndef MESH_COMPONENT_H
#define MESH_COMPONENT_H

#include "render_parts.hpp"
#include "vec3.h"
#include "vec2.h"
#include "bitmap.h"

#include <vector>
#include <array>

namespace RayZath::Engine::CPU
{
	struct RangedRay;
	struct TraversalResult;
}

namespace RayZath::Engine
{
	using Vertex = Math::vec3f;
	using Texcrd = Math::vec2f;
	using Normal = Math::vec3f;

	class Mesh;

	class Triangle
	{
	public:
		std::array<uint32_t, 3u> vertices;
		std::array<uint32_t, 3u> texcrds;
		std::array<uint32_t, 3u> normals;
		Math::vec3f normal;
		uint32_t material_id;


	public:
		Triangle(
			const std::array<uint32_t, 3u>& vs,
			const std::array<uint32_t, 3u>& ts,
			const std::array<uint32_t, 3u>& ns,
			const uint32_t& mat_id);


	public:
		void calculateNormal(const Mesh& mesh);
		BoundingBox boundingBox(const Mesh& mesh) const;

		bool areVertsValid() const;
		bool areTexcrdsValid() const;
		bool areNormalsValid() const;

		void closestIntersection(CPU::RangedRay& ray, CPU::TraversalResult& traversal, const Mesh& mesh) const;
		bool anyIntersection(const CPU::RangedRay& ray, Math::vec2f32& barycenter, const Mesh& mesh) const;
		Texcrd texcrdFromBarycenter(const Math::vec2f32 barycenter, const Mesh& mesh) const;
		Math::vec3f32 averageNormal(const Math::vec2f32 barycenter, const Mesh& mesh) const;
		void mapNormal(
			const Graphics::ColorF& map_color,
			Math::vec3f32& mapped_normal,
			const Mesh& mesh,
			const Math::vec3f32& scale) const;
	};
}

#endif