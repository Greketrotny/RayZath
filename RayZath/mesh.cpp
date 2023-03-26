#include "mesh.hpp"

#include <algorithm>
#include <functional>

#include <string>
#include <fstream>
#include <sstream>

namespace RayZath::Engine
{
	Mesh::Mesh(
		Updatable* parent,
		const ConStruct<Mesh>& conStruct)
		: WorldObject(parent, conStruct)
		, m_vertices(this, *this, conStruct.vertices)
		, m_texcrds(this, *this, conStruct.texcrds)
		, m_normals(this, *this, conStruct.normals)
		, m_triangles(this, *this, conStruct.triangles)
	{}

	ComponentContainer<Vertex>& Mesh::vertices()
	{
		return m_vertices;
	}
	ComponentContainer<Texcrd>& Mesh::texcrds()
	{
		return m_texcrds;
	}
	ComponentContainer<Normal>& Mesh::normals()
	{
		return m_normals;
	}
	ComponentContainer<Triangle>& Mesh::triangles()
	{
		return m_triangles;
	}
	const ComponentContainer<Vertex>& Mesh::vertices() const
	{
		return m_vertices;
	}
	const ComponentContainer<Texcrd>& Mesh::texcrds() const
	{
		return m_texcrds;
	}
	const ComponentContainer<Normal>& Mesh::normals() const
	{
		return m_normals;
	}
	const ComponentContainer<Triangle>& Mesh::triangles() const
	{
		return m_triangles;
	}

	uint32_t Mesh::createVertex(const Math::vec3f& vertex)
	{
		return m_vertices.add(vertex);
	}
	uint32_t Mesh::createVertex(const float& x, const float& y, const float& z)
	{
		return createVertex(Math::vec3f(x, y, z));
	}

	uint32_t Mesh::createTexcrd(const Math::vec2f& texcrd)
	{
		return m_texcrds.add(texcrd);
	}
	uint32_t Mesh::createTexcrd(const float& u, const float& v)
	{
		return createTexcrd(Math::vec2f(u, v));
	}

	uint32_t Mesh::createNormal(const Math::vec3f& normal)
	{
		return m_normals.add(normal.Normalized());
	}
	uint32_t Mesh::createNormal(const float& x, const float& y, const float& z)
	{
		return createNormal(Math::vec3f(x, y, z));
	}

	uint32_t Mesh::createTriangle(
		const std::array<uint32_t, 3u>& vs,
		const std::array<uint32_t, 3u>& ts,
		const std::array<uint32_t, 3u>& ns,
		const uint32_t& material_id)
	{/*
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
			return m_normals.sm_npos;*/

		return m_triangles.add(Triangle(vs, ts, ns, material_id));
	}
	uint32_t Mesh::createTriangle(
		const uint32_t& v1, const uint32_t& v2, const uint32_t& v3,
		const uint32_t& t1, const uint32_t& t2, const uint32_t& t3,
		const uint32_t& n1, const uint32_t& n2, const uint32_t& n3,
		const uint32_t& material_id)
	{
		return createTriangle({ v1, v2, v3 }, { t1, t2, t3 }, { n1, n2, n3 }, material_id);
	}

	void Mesh::reset()
	{
		m_triangles.removeAll();
		m_texcrds.removeAll();
		m_normals.removeAll();
		m_vertices.removeAll();
	}
	void Mesh::transform(const Engine::Transformation& transformation)
	{
		// vertices
		for (uint32_t i = 0; i < vertices().count(); i++)
		{
			auto& v = vertices()[i];
			v *= transformation.scale();
			v.RotateXYZ(transformation.rotation());
			v += transformation.position();
		}

		// normals
		for (uint32_t i = 0; i < normals().count(); i++)
		{
			auto& n = normals()[i];
			n /= transformation.scale();
			n.RotateXYZ(transformation.rotation());
			n.Normalize();
		}

		stateRegister().RequestUpdate();
	}

	void Mesh::update()
	{
		if (!stateRegister().RequiresUpdate()) return;

		m_triangles.update();
		for (uint32_t i = 0u; i < m_triangles.count(); ++i)
		{
			m_triangles[i].calculateNormal(*this);
		}

		stateRegister().update();
	}
}