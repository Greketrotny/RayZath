#include "mesh_structure.h"

#include <algorithm>
#include <functional>

#include <string>
#include <fstream>
#include <strstream>
#include <sstream>

namespace RayZath::Engine
{
	// ~~~~~~~~ [STRUCT] MeshStructure ~~~~~~~~
	MeshStructure::MeshStructure(
		Updatable* parent,
		const ConStruct<MeshStructure>& conStruct)
		: WorldObject(parent, conStruct)
		, m_vertices(this, *this, conStruct.vertices)
		, m_texcrds(this, *this, conStruct.texcrds)
		, m_normals(this, *this, conStruct.normals)
		, m_triangles(this, *this, conStruct.triangles)
	{}

	uint32_t MeshStructure::CreateVertex(const Math::vec3f& vertex)
	{
		return m_vertices.Add(vertex);
	}
	uint32_t MeshStructure::CreateVertex(const float& x, const float& y, const float& z)
	{
		return CreateVertex(Math::vec3f(x, y, z));
	}

	uint32_t MeshStructure::CreateTexcrd(const Math::vec2f& texcrd)
	{
		return m_texcrds.Add(texcrd);
	}
	uint32_t MeshStructure::CreateTexcrd(const float& u, const float& v)
	{
		return CreateTexcrd(Math::vec2f(u, v));
	}

	uint32_t MeshStructure::CreateNormal(const Math::vec3f& normal)
	{
		return m_normals.Add(normal.Normalized());
	}
	uint32_t MeshStructure::CreateNormal(const float& x, const float& y, const float& z)
	{
		return CreateNormal(Math::vec3f(x, y, z));
	}

	uint32_t MeshStructure::CreateTriangle(
		const std::array<uint32_t, 3u>& vs,
		const std::array<uint32_t, 3u>& ts,
		const std::array<uint32_t, 3u>& ns,
		const uint32_t& material_id)
	{
		if (vs[0] >= m_vertices.GetCount() ||
			vs[1] >= m_vertices.GetCount() ||
			vs[2] >= m_vertices.GetCount())
			return m_vertices.sm_npos;

		if ((ts[0] >= m_texcrds.GetCount() && ts[0] != m_texcrds.sm_npos) ||
			(ts[1] >= m_texcrds.GetCount() && ts[1] != m_texcrds.sm_npos) ||
			(ts[2] >= m_texcrds.GetCount() && ts[2] != m_texcrds.sm_npos))
			return m_texcrds.sm_npos;

		if ((ns[0] >= m_normals.GetCount() && ns[0] != m_texcrds.sm_npos) ||
			(ns[1] >= m_normals.GetCount() && ns[1] != m_texcrds.sm_npos) ||
			(ns[2] >= m_normals.GetCount() && ns[2] != m_texcrds.sm_npos))
			return m_normals.sm_npos;

		return m_triangles.Add(Triangle(vs, ts, ns, material_id));
	}
	uint32_t MeshStructure::CreateTriangle(
		const uint32_t& v1, const uint32_t& v2, const uint32_t& v3,
		const uint32_t& t1, const uint32_t& t2, const uint32_t& t3,
		const uint32_t& n1, const uint32_t& n2, const uint32_t& n3,
		const uint32_t& material_id)
	{
		return CreateTriangle({ v1, v2, v3 }, { t1, t2, t3 }, { n1, n2, n3 }, material_id);
	}

	void MeshStructure::Reset()
	{
		m_triangles.RemoveAll();
		m_texcrds.RemoveAll();
		m_normals.RemoveAll();
		m_vertices.RemoveAll();
	}

	ComponentContainer<Vertex>& MeshStructure::GetVertices()
	{
		return m_vertices;
	}
	ComponentContainer<Texcrd>& MeshStructure::GetTexcrds()
	{
		return m_texcrds;
	}
	ComponentContainer<Normal>& MeshStructure::GetNormals()
	{
		return m_normals;
	}
	ComponentContainer<Triangle>& MeshStructure::GetTriangles()
	{
		return m_triangles;
	}
	const ComponentContainer<Vertex>& MeshStructure::GetVertices() const
	{
		return m_vertices;
	}
	const ComponentContainer<Texcrd>& MeshStructure::GetTexcrds() const
	{
		return m_texcrds;
	}
	const ComponentContainer<Normal>& MeshStructure::GetNormals() const
	{
		return m_normals;
	}
	const ComponentContainer<Triangle>& MeshStructure::GetTriangles() const
	{
		return m_triangles;
	}

	void MeshStructure::Update()
	{
		if (!GetStateRegister().RequiresUpdate()) return;

		m_triangles.Update();
		for (unsigned int i = 0u; i < m_triangles.GetCount(); ++i)
		{
			m_triangles[i].CalculateNormal(*this);
		}

		GetStateRegister().Update();
	}
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
}