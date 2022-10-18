#ifndef MESH_COMPONENT_H
#define MESH_COMPONENT_H

#include "render_parts.hpp"
#include "vec3.h"
#include "vec2.h"

#include <vector>
#include <array>

namespace RayZath::Engine
{
	using Vertex = Math::vec3f;
	using Texcrd = Math::vec2f;
	using Normal = Math::vec3f;

	struct Mesh;

	struct Triangle
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
	};
}

#endif