#ifndef MESH_STRUCTURE_H
#define MESH_STRUCTURE_H

#include "component_container.h"
#include "mesh_component.h"

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
		MeshStructure(
			Updatable* updatable,
			const ConStruct<MeshStructure>& con_struct);


	public:
		uint32_t CreateVertex(const Math::vec3f& vertex);
		uint32_t CreateVertex(const float& x, const float& y, const float& z);

		uint32_t CreateTexcrd(const Math::vec2f& texcrd);
		uint32_t CreateTexcrd(const float& u, const float& v);

		uint32_t CreateNormal(const Math::vec3f& normal);
		uint32_t CreateNormal(const float& x, const float& y, const float& z);

		uint32_t CreateTriangle(
			const uint32_t& v1, const uint32_t& v2, const uint32_t& v3,
			const uint32_t& t1, const uint32_t& t2, const uint32_t& t3,
			const uint32_t& n1, const uint32_t& n2, const uint32_t& n3,
			const uint32_t& material_id = 0u);
		uint32_t CreateTriangle(
			const std::array<uint32_t, 3u>& vs,
			const std::array<uint32_t, 3u>& ts = {
				ComponentContainer<Texcrd>::sm_npos,
				ComponentContainer<Texcrd>::sm_npos,
				ComponentContainer<Texcrd>::sm_npos
			},
			const std::array<uint32_t, 3u>& ns = {
				ComponentContainer<Normal>::sm_npos,
				ComponentContainer<Normal>::sm_npos,
				ComponentContainer<Normal>::sm_npos
			},
			const uint32_t& material_id = 0u);


		void Reset();

		ComponentContainer<Vertex>& GetVertices();
		ComponentContainer<Texcrd>& GetTexcrds();
		ComponentContainer<Normal>& GetNormals();
		ComponentContainer<Triangle>& GetTriangles();
		const ComponentContainer<Vertex>& GetVertices() const;
		const ComponentContainer<Texcrd>& GetTexcrds() const;
		const ComponentContainer<Normal>& GetNormals() const;
		const ComponentContainer<Triangle>& GetTriangles() const;

		void Update() override;
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
