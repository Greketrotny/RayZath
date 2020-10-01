#include "mesh.h"

#include <fstream>
#include <algorithm>

namespace RayZath
{
	// ~~~~~~~~ [STRUCT] MeshStructure ~~~~~~~~
	MeshStructure::MeshStructure(Updatable* parent)
		: Updatable(parent)
	{}
	MeshStructure::MeshStructure(
		Updatable* parent,
		const uint32_t& vertices,
		const uint32_t& texcrds,
		const uint32_t& triangles)
		: Updatable(parent)
		, m_vertices(vertices)
		, m_texcrds(texcrds)
		, m_triangles(triangles)
	{}
	MeshStructure::~MeshStructure()
	{}

	Vertex* MeshStructure::CreateVertex(const Math::vec3<float>& vertex)
	{
		return m_vertices.Add(vertex);
	}
	Vertex* MeshStructure::CreateVertex(const float& x, const float& y, const float& z)
	{
		return CreateVertex(Math::vec3<float>(x, y, z));
	}
	
	Texcrd* MeshStructure::CreateTexcrd(const Texcrd& texcrd)
	{
		return m_texcrds.Add(texcrd);
	}
	Texcrd* MeshStructure::CreateTexcrd(const float& u, const float& v)
	{
		return CreateTexcrd(Texcrd(u, v));
	}
	
	bool MeshStructure::CreateTriangle(
		const uint32_t& v1, const uint32_t& v2, const uint32_t& v3,
		const uint32_t& t1, const uint32_t& t2, const uint32_t& t3,
		const Graphics::Color& color)
	{
		if (v1 >= m_vertices.GetCount() ||
			v2 >= m_vertices.GetCount() ||
			v3 >= m_vertices.GetCount())
			return false;

		if (t1 >= m_texcrds.GetCount() ||
			t2 >= m_texcrds.GetCount() ||
			t3 >= m_texcrds.GetCount())
			return false;

		return (m_triangles.Add(Triangle(
			&m_vertices[v1], &m_vertices[v2], &m_vertices[v3],
			&m_texcrds[t1], &m_texcrds[t2], &m_texcrds[t3],
			color)) != nullptr);
	}
	bool MeshStructure::CreateTriangle(
		Vertex* v1, Vertex* v2, Vertex* v3,
		Texcrd* t1, Texcrd* t2, Texcrd* t3,
		const Graphics::Color& color)
	{
		if ((v1 - &m_vertices[0]) >= m_vertices.GetCount() ||
			(v2 - &m_vertices[0]) >= m_vertices.GetCount() ||
			(v3 - &m_vertices[0]) >= m_vertices.GetCount())
			return false;

		if (t1 != nullptr && t2 != nullptr && t3 != nullptr)
		{
			if ((t1 - &m_texcrds[0]) >= m_texcrds.GetCount() ||
				(t2 - &m_texcrds[0]) >= m_texcrds.GetCount() ||
				(t3 - &m_texcrds[0]) >= m_texcrds.GetCount())
				return false;
		}

		return (m_triangles.Add(Triangle(
			v1, v2, v3,
			t1, t2, t3,
			color)) != nullptr);
	}
	
	void MeshStructure::Reset()
	{
		m_triangles.Reset();
		m_texcrds.Reset();
		m_vertices.Reset();
	}
	void MeshStructure::Reset(
		const uint32_t& vertices_capacity,
		const uint32_t& texcrds_capacity,
		const uint32_t& triangles_capacity)
	{
		Reset();

		m_vertices.Resize(vertices_capacity);
		m_texcrds.Resize(texcrds_capacity);
		m_triangles.Resize(triangles_capacity);
	}

	ComponentContainer<Math::vec3<float>>& MeshStructure::GetVertices()
	{
		return m_vertices;
	}
	ComponentContainer<Texcrd>& MeshStructure::GetTexcrds()
	{
		return m_texcrds;
	}
	ComponentContainer<Triangle>& MeshStructure::GetTriangles()
	{
		return m_triangles;
	}
	const ComponentContainer<Math::vec3<float>>& MeshStructure::GetVertices() const
	{
		return m_vertices;
	}
	const ComponentContainer<Texcrd>& MeshStructure::GetTexcrds() const
	{
		return m_texcrds;
	}
	const ComponentContainer<Triangle>& MeshStructure::GetTriangles() const
	{
		return m_triangles;
	}

	void MeshStructure::Update()
	{
		//if (!GetStateRegister().RequiresUpdate()) return;

		// update triangles
		m_triangles.Update();
		for (unsigned int i = 0u; i < m_triangles.GetCount(); ++i)
		{
			m_triangles[i].CalculateNormal();
		}

		GetStateRegister().Update();
	}
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


	// ~~~~~~~~ [CLASS] Mesh ~~~~~~~~
	Mesh::Mesh(
		const uint32_t& id,
		Updatable* updatable,
		const ConStruct<Mesh>& conStruct)
		: RenderObject(id, updatable, conStruct)
		, m_mesh_data(
			this,
			conStruct.vertices_capacity, 
			conStruct.texcrds_capacity, 
			conStruct.triangles_capacity)
	{}
	Mesh::~Mesh()
	{
		UnloadTexture();
	}

	void Mesh::LoadTexture(const Texture& newTexture)
	{
		if (this->m_pTexture == nullptr) this->m_pTexture = new Texture(newTexture);
		else *this->m_pTexture = newTexture;

		GetStateRegister().MakeModified();
	}
	void Mesh::UnloadTexture()
	{
		if (m_pTexture)
		{
			delete m_pTexture;
			m_pTexture = nullptr;

			GetStateRegister().MakeModified();
		}
	}
	
	const Texture* Mesh::GetTexture() const
	{
		return m_pTexture;
	}
	MeshStructure& Mesh::GetMeshStructure()
	{
		return m_mesh_data;
	}
	const MeshStructure& Mesh::GetMeshStructure() const
	{
		return m_mesh_data;
	}

	void Mesh::Update()
	{
		//if (!GetStateRegister().RequiresUpdate()) return;

		m_mesh_data.Update();
		CalculateBoundingBox();

		GetStateRegister().Update();
	}

	void Mesh::CalculateBoundingBox()
	{
		// [>] Update bounding volume
		m_bounding_box.Reset();

		// setup bounding planes
		Math::vec3<float> P[6];
		Math::vec3<float> vN[6];

		vN[0] = Math::vec3<float>(0.0f, 1.0f, 0.0f);	// top
		vN[1] = Math::vec3<float>(0.0f, -1.0f, 0.0f);	// bottom
		vN[2] = Math::vec3<float>(-1.0f, 0.0f, 0.0f);	// left
		vN[3] = Math::vec3<float>(1.0f, 0.0f, 0.0f);	// right
		vN[4] = Math::vec3<float>(0.0f, 0.0f, -1.0f);	// front
		vN[5] = Math::vec3<float>(0.0f, 0.0f, 1.0f);	// back

		// rotate planes' normals
		for (int i = 0; i < 6; i++)
		{
			vN[i].RotateZYX(-m_rotation);
		}

		// expand planes by each farthest (for plane direction) vertex
		auto& vertices = m_mesh_data.GetVertices();
		for (unsigned int i = 0u; i < vertices.GetCount(); ++i)
		{
			Math::vec3<float> V = vertices[i];
			V += m_center;
			V *= m_scale;

			for (int j = 0; j < 6; j++)
			{
				if (Math::vec3<float>::DotProduct(V - P[j], vN[j]) > 0.0f)
					P[j] = V;
			}
		}

		// rotate planes back
		for (int i = 0; i < 6; i++)
		{
			P[i].RotateXYZ(m_rotation);
		}

		// set bounding box extents
		m_bounding_box.min.x = P[2].x;
		m_bounding_box.min.y = P[1].y;
		m_bounding_box.min.z = P[4].z;
		m_bounding_box.max.x = P[3].x;
		m_bounding_box.max.y = P[0].y;
		m_bounding_box.max.z = P[5].z;

		// transpose extents by object position
		m_bounding_box.min += m_position;
		m_bounding_box.max += m_position;
	}
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
}