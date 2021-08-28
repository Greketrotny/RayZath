#include "mesh_structure.h"

#include <algorithm>
#include <functional>

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
		, m_vertices(this, *this, conStruct.vertices)
		, m_texcrds(this, *this, conStruct.texcrds)
		, m_normals(this, *this, conStruct.normals)
		, m_triangles(this, *this, conStruct.triangles)
	{}

	bool MeshStructure::LoadFromFile(const std::string& file_name)
	{
		// [>] Open specified file
		std::ifstream ifs;
		ifs.open(file_name, std::ios_base::in);
		if (!ifs.is_open()) return false;

		Reset();
		static constexpr uint32_t max_n_gon = 8u;

		// [>] Read file and construct mesh
		std::string file_line, type;
		while (std::getline(ifs, file_line))
		{
			if (file_line.empty()) continue;
			while (file_line.back() == ' ') file_line.pop_back();

			std::stringstream ss(file_line);
			ss >> type;

			if (type == "v")
			{
				Math::vec3f v;
				ss >> v.x >> v.y >> v.z;
				CreateVertex(v);
			}
			else if (type == "vt")
			{
				Math::vec2f t;
				ss >> t.x >> t.y;
				CreateTexcrd(t);
			}
			else if (type == "vn")
			{
				Math::vec3f n;
				ss >> n.x >> n.y >> n.z;
				CreateNormal(n);
			}
			else if (type == "f")
			{
				// extract vertices data to separate strings
				std::string vertex_as_string[max_n_gon];
				uint8_t face_v_count = 0u;
				while (!ss.eof() && face_v_count < max_n_gon)
				{
					ss >> vertex_as_string[face_v_count];
					face_v_count++;
				}

				// allocate vertex data buffers
				uint32_t v[max_n_gon];
				uint32_t t[max_n_gon];
				uint32_t n[max_n_gon];
				for (uint32_t i = 0u; i < max_n_gon; i++)
				{
					v[i] = m_vertices.sm_npos;
					t[i] = m_texcrds.sm_npos;
					n[i] = m_normals.sm_npos;
				}


				for (uint8_t vertex_idx = 0u; vertex_idx < face_v_count; vertex_idx++)
				{
					const std::string& vertex_desc = vertex_as_string[vertex_idx];
					std::vector<std::string> indices(3u);
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
							v[vertex_idx] = vp_idx - 1;
						}
						else if (vp_idx < 0 && int32_t(m_vertices.GetCount()) + vp_idx >= 0)
						{
							v[vertex_idx] = m_vertices.GetCount() + vp_idx;
						}
					}

					// vertex texcrd
					if (!indices[1].empty())
					{
						int32_t vt_idx = std::stoi(indices[1]);
						if (vt_idx > 0 && vt_idx <= int32_t(m_texcrds.GetCount()))
						{
							t[vertex_idx] = vt_idx - 1;
						}
						else if (vt_idx < 0 && int32_t(m_texcrds.GetCount()) + vt_idx >= 0)
						{
							t[vertex_idx] = m_texcrds.GetCount() + vt_idx;
						}
					}

					// vertex normal
					if (!indices[2].empty())
					{
						int32_t vn_idx = std::stoi(indices[2]);
						if (vn_idx > 0 && vn_idx <= int32_t(m_normals.GetCount()))
						{
							n[vertex_idx] = vn_idx - 1;
						}
						else if (vn_idx < 0 && int32_t(m_normals.GetCount()) + vn_idx >= 0)
						{
							n[vertex_idx] = m_normals.GetCount() + vn_idx;
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