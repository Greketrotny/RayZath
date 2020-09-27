#include "mesh.h"

#include <fstream>
#include <algorithm>

namespace RayZath
{
	// ~~~~~~~~ [STRUCT] MeshData ~~~~~~~~
	MeshData::MeshData()
	{}
	MeshData::MeshData(
		const uint32_t& vertices,
		const uint32_t& texcrds,
		const uint32_t& triangles)
		: m_vertices(vertices)
		, m_texcrds(texcrds)
		, m_triangles(triangles)
	{}
	MeshData::~MeshData()
	{}

	Vertex* MeshData::CreateVertex(const Math::vec3<float>& vertex)
	{
		return m_vertices.Add(vertex);
	}
	Vertex* MeshData::CreateVertex(const float& x, const float& y, const float& z)
	{
		return CreateVertex(Math::vec3<float>(x, y, z));
	}
	
	Texcrd* MeshData::CreateTexcrd(const Texcrd& texcrd)
	{
		return m_texcrds.Add(texcrd);
	}
	Texcrd* MeshData::CreateTexcrd(const float& u, const float& v)
	{
		return CreateTexcrd(Texcrd(u, v));
	}
	
	bool MeshData::CreateTriangle(
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
	bool MeshData::CreateTriangle(
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
	
	void MeshData::Reset()
	{
		m_triangles.Reset();
		m_texcrds.Reset();
		m_vertices.Reset();
	}
	void MeshData::Reset(
		const uint32_t& vertices_capacity,
		const uint32_t& texcrds_capacity,
		const uint32_t& triangles_capacity)
	{
		Reset();

		m_vertices.Resize(vertices_capacity);
		m_texcrds.Resize(texcrds_capacity);
		m_triangles.Resize(triangles_capacity);
	}

	ComponentStorage<Math::vec3<float>>& MeshData::GetVertices()
	{
		return m_vertices;
	}
	ComponentStorage<Texcrd>& MeshData::GetTexcrds()
	{
		return m_texcrds;
	}
	ComponentStorage<Triangle>& MeshData::GetTriangles()
	{
		return m_triangles;
	}
	const ComponentStorage<Math::vec3<float>>& MeshData::GetVertices() const
	{
		return m_vertices;
	}
	const ComponentStorage<Texcrd>& MeshData::GetTexcrds() const
	{
		return m_texcrds;
	}
	const ComponentStorage<Triangle>& MeshData::GetTriangles() const
	{
		return m_triangles;
	}
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


	// ~~~~~~~~ [CLASS] Mesh ~~~~~~~~
	Mesh::Mesh(
		const size_t& id,
		Updatable* updatable,
		const ConStruct<Mesh>& conStruct)
		: RenderObject(id, updatable, conStruct)
		, m_mesh_data(
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
	void Mesh::TransposeComponents()
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


		// [>] Update triangles
		auto& triangles = m_mesh_data.GetTriangles();
		for (unsigned int i = 0u; i < triangles.GetCount(); ++i)
		{
			triangles[i].CalculateNormal();
		}
	}
	
	const Texture* Mesh::GetTexture() const
	{
		return m_pTexture;
	}
	MeshData& Mesh::GetMeshData()
	{
		return m_mesh_data;
	}
	const MeshData& Mesh::GetMeshData() const
	{
		return m_mesh_data;
	}

	void Mesh::Update()
	{
		TransposeComponents();
	}
	// --------------------------------------------|
}