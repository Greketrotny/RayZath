#include "mesh.h"

#include <algorithm>

#include <string>
#include <fstream>
#include <strstream>
#include <sstream>

namespace RayZath
{
	// ~~~~~~~~ [STRUCT] MeshStructure ~~~~~~~~
	MeshStructure::MeshStructure(
		Updatable* parent,
		const ConStruct<MeshStructure>& conStruct)
		: Updatable(parent)
		, m_vertices(this, conStruct.vertices)
		, m_texcrds(this, conStruct.texcrds)
		, m_normals(this, conStruct.normals)
		, m_triangles(this, conStruct.triangles)
	{}
	MeshStructure::~MeshStructure()
	{}

	bool MeshStructure::LoadFromFile(const std::wstring& file_name)
	{
		// [>] Open specified file
		std::wifstream ifs;
		ifs.open(file_name, std::ios_base::in);
		if (!ifs.is_open()) return false;
		

		// [>] Count mesh components
		uint32_t v_count = 0u;
		uint32_t vt_count = 0u;
		uint32_t vn_count = 0u;
		uint32_t f_count = 0u;

		std::wstring file_line, type;
		while (std::getline(ifs, file_line))
		{
			if (file_line.empty()) continue;

			std::wstringstream ss(file_line);
			ss >> type;

			if (type == L"v") v_count++;
			else if (type == L"vt") vt_count++;
			else if (type == L"vn") vn_count++;
			else if (type == L"f") f_count++;
		}


		// [>] Reset mesh structure
		ifs.clear();
		ifs.seekg(0u);
		Reset(v_count, vt_count, vn_count, f_count);

		static constexpr uint32_t max_n_gon = 8u;


		// [>] Read file and construct mesh
		while (std::getline(ifs, file_line))
		{
			if (file_line.empty()) continue;
			while (file_line.back() == ' ') file_line.pop_back();

			std::wstringstream ss(file_line);
			ss >> type;

			if (type == L"v")
			{
				Math::vec3f v;
				ss >> v.x >> v.y >> v.z;
				CreateVertex(v);
			}
			else if (type == L"vt")
			{
				Texcrd t;
				ss >> t.u >> t.v;
				CreateTexcrd(t);
			}
			else if (type == L"vn")
			{
				Normal n;
				ss >> n.x >> n.y >> n.z;
				CreateNormal(n);
			}
			else if (type == L"f")
			{
				// extract vertices data to separate strings
				std::wstring vertex_as_string[max_n_gon];
				uint8_t face_v_count = 0u;
				while (!ss.eof() && face_v_count < max_n_gon)
				{
					ss >> vertex_as_string[face_v_count];
					face_v_count++;
				}

				// allocate vertex data buffers
				Vertex* v[max_n_gon];
				Texcrd* t[max_n_gon];
				Math::vec3f* n[max_n_gon];
				for (uint32_t i = 0u; i < max_n_gon; i++)
				{
					v[i] = nullptr;
					t[i] = nullptr;
					n[i] = nullptr;
				}


				for (uint8_t vertex_idx = 0u; vertex_idx < face_v_count; vertex_idx++)
				{
					std::wstring vertex_desc = vertex_as_string[vertex_idx];
					std::vector<std::wstring> indices(3u);
					for (size_t i = 0u, c = 0u; i < 3u && c < vertex_desc.size(); c++)
					{
						if (vertex_desc[c] == L'/') i++;
						else indices[i] += vertex_desc[c];
					}

					// vertex position
					if (!indices[0].empty())
					{
						int32_t vp_idx = std::stoi(indices[0]);
						if (vp_idx > 0 && vp_idx <= int32_t(m_vertices.GetCount()))
						{
							v[vertex_idx] = &m_vertices[vp_idx - 1];
						}
						else if (vp_idx < 0 && int32_t(m_vertices.GetCount()) + vp_idx >= 0)
						{
							v[vertex_idx] = &m_vertices[int32_t(m_vertices.GetCount()) + vp_idx];
						}
					}

					// vertex texcrd
					if (!indices[1].empty())
					{
						int32_t vt_idx = std::stoi(indices[1]);
						if (vt_idx > 0 && vt_idx <= m_texcrds.GetCount())
						{
							t[vertex_idx] = &m_texcrds[vt_idx - 1];
						}
						else if (vt_idx < 0 && m_texcrds.GetCount() + vt_idx >= 0)
						{
							t[vertex_idx] = &m_texcrds[m_texcrds.GetCount() + vt_idx];
						}
					}

					// vertex normal
					if (!indices[2].empty())
					{
						int32_t vn_idx = std::stoi(indices[2]);
						if (vn_idx > 0 && vn_idx <= m_normals.GetCount())
						{
							n[vertex_idx] = &m_normals[vn_idx - 1];
						}
						else if (vn_idx < 0 && m_normals.GetCount() + vn_idx >= 0)
						{
							n[vertex_idx] = &m_normals[m_normals.GetCount() + vn_idx];
						}
					}
				}

				// create face
				if (face_v_count == 3u)
				{	// triangle

					CreateTriangle(
						v[0], v[1], v[2],
						t[0], t[1], t[2],
						n[0], n[1], n[2]);
				}
				else if (face_v_count == 4u)
				{	// quadrilateral

					// for now just split quad into two touching triangles
					CreateTriangle(
						v[0], v[1], v[2],
						t[0], t[1], t[2],
						n[0], n[1], n[2]);

					CreateTriangle(
						v[0], v[2], v[3],
						t[0], t[2], t[3],
						n[0], n[2], n[3]);
				}
				else
				{	// polygon (tesselate into triangles)
					for (uint8_t i = 1u; i < face_v_count - 1u; i++)
					{
						CreateTriangle(
							v[0], v[i], v[i + 1u],
							t[0], t[i], t[i + 1u],
							n[0], n[i], n[i + 1u]);
					}
				}
			}
		}

		ifs.close();

		GetStateRegister().RequestUpdate();
		return true;
	}

	Vertex* MeshStructure::CreateVertex(const Math::vec3f& vertex)
	{
		return m_vertices.Add(vertex);
	}
	Vertex* MeshStructure::CreateVertex(const float& x, const float& y, const float& z)
	{
		return CreateVertex(Math::vec3f(x, y, z));
	}

	Texcrd* MeshStructure::CreateTexcrd(const Texcrd& texcrd)
	{
		return m_texcrds.Add(texcrd);
	}
	Texcrd* MeshStructure::CreateTexcrd(const float& u, const float& v)
	{
		return CreateTexcrd(Texcrd(u, v));
	}

	Math::vec3f* MeshStructure::CreateNormal(const Math::vec3f& normal)
	{
		return m_normals.Add(Math::vec3f::Normalize(normal));
	}
	Math::vec3f* MeshStructure::CreateNormal(const float& x, const float& y, const float& z)
	{
		return CreateNormal(Math::vec3f(x, y, z));
	}

	bool MeshStructure::CreateTriangle(
		const uint32_t& v1, const uint32_t& v2, const uint32_t& v3,
		const uint32_t& t1, const uint32_t& t2, const uint32_t& t3,
		const uint32_t& n1, const uint32_t& n2, const uint32_t& n3,
		const uint32_t& material_id)
	{
		if (v1 >= m_vertices.GetCount() ||
			v2 >= m_vertices.GetCount() ||
			v3 >= m_vertices.GetCount())
			return false;

		if (t1 >= m_texcrds.GetCount() ||
			t2 >= m_texcrds.GetCount() ||
			t3 >= m_texcrds.GetCount())
			return false;

		if (n1 >= m_normals.GetCount() ||
			n2 >= m_normals.GetCount() ||
			n3 >= m_normals.GetCount())
			return false;

		return (m_triangles.Add(Triangle(
			&m_vertices[v1], &m_vertices[v2], &m_vertices[v3],
			&m_texcrds[t1], &m_texcrds[t2], &m_texcrds[t3],
			&m_normals[n1], &m_normals[n2], &m_normals[n3],
			material_id)) != nullptr);
	}
	bool MeshStructure::CreateTriangle(
		Vertex* v1, Vertex* v2, Vertex* v3,
		Texcrd* t1, Texcrd* t2, Texcrd* t3,
		Normal* n1, Normal* n2, Normal* n3,
		const uint32_t& material_id)
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

		if (n1 != nullptr && n2 != nullptr && n3 != nullptr)
		{
			if ((n1 - &m_normals[0]) >= m_normals.GetCount() ||
				(n2 - &m_normals[0]) >= m_normals.GetCount() ||
				(n3 - &m_normals[0]) >= m_normals.GetCount())
				return false;
		}

		return (m_triangles.Add(Triangle(
			v1, v2, v3,
			t1, t2, t3,
			n1, n2, n3,
			material_id)) != nullptr);
	}

	void MeshStructure::Reset()
	{
		m_triangles.Reset();
		m_texcrds.Reset();
		m_normals.Reset();
		m_vertices.Reset();
	}
	void MeshStructure::Reset(
		const uint32_t& vertices_capacity,
		const uint32_t& texcrds_capacity,
		const uint32_t& normals_capacity,
		const uint32_t& triangles_capacity)
	{
		m_triangles.Reset(triangles_capacity);
		m_vertices.Reset(vertices_capacity);
		m_texcrds.Reset(texcrds_capacity);
		m_normals.Reset(normals_capacity);
	}

	ComponentContainer<Math::vec3f>& MeshStructure::GetVertices()
	{
		return m_vertices;
	}
	ComponentContainer<Texcrd>& MeshStructure::GetTexcrds()
	{
		return m_texcrds;
	}
	ComponentContainer<Math::vec3f>& MeshStructure::GetNormals()
	{
		return m_normals;
	}
	ComponentContainer<Triangle>& MeshStructure::GetTriangles()
	{
		return m_triangles;
	}
	const ComponentContainer<Math::vec3f>& MeshStructure::GetVertices() const
	{
		return m_vertices;
	}
	const ComponentContainer<Texcrd>& MeshStructure::GetTexcrds() const
	{
		return m_texcrds;
	}
	const ComponentContainer<Math::vec3f>& MeshStructure::GetNormals() const
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
		Updatable* updatable,
		const ConStruct<Mesh>& conStruct)
		: RenderObject(updatable, conStruct)
		, m_mesh_structure(conStruct.mesh_structure, std::bind(&Mesh::NotifyMeshStructure, this))
	{
		for (uint32_t i = 0u; i < sm_mat_count; i++)
			m_materials->SetNotifyFunction(std::bind(&Mesh::NotifyMaterial, this));

		RZAssert(bool(conStruct.material), L"handle to material was nullptr");
		SetMaterial(conStruct.material, 0u);
	}
	Mesh::~Mesh()
	{
	}


	void Mesh::SetMeshStructure(const Handle<MeshStructure>& mesh_structure)
	{
		m_mesh_structure = mesh_structure;
		GetStateRegister().RequestUpdate();
	}
	void  Mesh::SetMaterial(
		const Handle<Material>& material,
		const uint32_t& material_index)
	{
		if (material_index >= GetMaterialCount()) return;
		if (!material) return;

		m_materials[material_index] = material;
		GetStateRegister().RequestUpdate();
	}

	const Handle<MeshStructure>& Mesh::GetMeshStructure() const
	{
		return static_cast<const Handle<MeshStructure>&>(m_mesh_structure);
	}
	const Handle<Material>& Mesh::GetMaterial(const uint32_t& material_index) const
	{
		if (material_index >= GetMaterialCount()) return Handle<Material>();
		return m_materials[material_index];
	}

	void Mesh::Update()
	{
		if (!GetStateRegister().RequiresUpdate()) return;

		CalculateBoundingBox();

		GetStateRegister().Update();
	}
	void Mesh::NotifyMeshStructure()
	{
		GetStateRegister().MakeModified();
	}
	void Mesh::NotifyMaterial()
	{
		GetStateRegister().MakeModified();
	}

	void Mesh::CalculateBoundingBox()
	{
		// [>] Update bounding volume
		m_bounding_box.Reset();

		if (!m_mesh_structure) return;

		// setup bounding planes
		Math::vec3f P[6];
		Math::vec3f vN[6];

		vN[0] = Math::vec3f(0.0f, 1.0f, 0.0f);	// top
		vN[1] = Math::vec3f(0.0f, -1.0f, 0.0f);	// bottom
		vN[2] = Math::vec3f(-1.0f, 0.0f, 0.0f);	// left
		vN[3] = Math::vec3f(1.0f, 0.0f, 0.0f);	// right
		vN[4] = Math::vec3f(0.0f, 0.0f, -1.0f);	// front
		vN[5] = Math::vec3f(0.0f, 0.0f, 1.0f);	// back

		// rotate planes' normals
		for (int i = 0; i < 6; i++)
		{
			vN[i].RotateZYX(-m_rotation);
		}

		// expand planes by each farthest (for plane direction) vertex
		auto& vertices = m_mesh_structure->GetVertices();
		for (unsigned int i = 0u; i < vertices.GetCount(); ++i)
		{
			Math::vec3f V = vertices[i];
			V += m_center;
			V *= m_scale;

			for (int j = 0; j < 6; j++)
			{
				if (Math::vec3f::DotProduct(V - P[j], vN[j]) > 0.0f)
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