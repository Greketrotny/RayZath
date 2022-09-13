#ifndef MESH_STRUCTURE_H
#define MESH_STRUCTURE_H

#include "component_container.hpp"
#include "mesh_component.hpp"

#include <array>

namespace RayZath::Engine
{
	struct MeshStructure;
	template <> struct ConStruct<MeshStructure>;

	struct MeshStructure
		: public WorldObject
	{
	private:
		ComponentContainer<Vertex> m_vertices;
		ComponentContainer<Texcrd> m_texcrds;
		ComponentContainer<Normal> m_normals;
		ComponentContainer<Triangle> m_triangles;
	public:
		using triple_index_t = std::array<uint32_t, 3u>;
		static constexpr triple_index_t ids_unused{
				ComponentContainer<Texcrd>::sm_npos,
				ComponentContainer<Texcrd>::sm_npos,
				ComponentContainer<Texcrd>::sm_npos };


	public:
		MeshStructure(
			Updatable* updatable,
			const ConStruct<MeshStructure>& con_struct);


	public:
		uint32_t createVertex(const Math::vec3f& vertex);
		uint32_t createVertex(const float& x, const float& y, const float& z);

		uint32_t createTexcrd(const Math::vec2f& texcrd);
		uint32_t createTexcrd(const float& u, const float& v);

		uint32_t createNormal(const Math::vec3f& normal);
		uint32_t createNormal(const float& x, const float& y, const float& z);

		uint32_t createTriangle(
			const uint32_t& v1, const uint32_t& v2, const uint32_t& v3,
			const uint32_t& t1, const uint32_t& t2, const uint32_t& t3,
			const uint32_t& n1, const uint32_t& n2, const uint32_t& n3,
			const uint32_t& material_id = 0u);
		uint32_t createTriangle(
			const triple_index_t& vs,
			const triple_index_t& ts = ids_unused,
			const triple_index_t& ns = ids_unused,
			const uint32_t& material_id = 0u);


		void reset();

		ComponentContainer<Vertex>& vertices();
		ComponentContainer<Texcrd>& texcrds();
		ComponentContainer<Normal>& normals();
		ComponentContainer<Triangle>& triangles();
		const ComponentContainer<Vertex>& vertices() const;
		const ComponentContainer<Texcrd>& texcrds() const;
		const ComponentContainer<Normal>& normals() const;
		const ComponentContainer<Triangle>& triangles() const;

		void update() override;
	};
	template <> struct ConStruct<MeshStructure>
		: public ConStruct<WorldObject>
	{
		uint32_t vertices, texcrds, normals, triangles;

		ConStruct(
			const std::string& name = "mesh structure",
			const uint32_t& vertices = 2u,
			const uint32_t& texcrds = 2u,
			const uint32_t& normals = 2u,
			const uint32_t& triangles = 2u)
			: ConStruct<WorldObject>(name)
			, vertices(vertices)
			, texcrds(texcrds)
			, normals(normals)
			, triangles(triangles)
		{}
	};
}

#endif
