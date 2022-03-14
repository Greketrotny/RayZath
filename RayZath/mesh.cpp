#include "mesh.h"

namespace RayZath::Engine
{
	// ~~~~~~~~ [CLASS] Mesh ~~~~~~~~
	Mesh::Mesh(
		Updatable* updatable,
		const ConStruct<Mesh>& conStruct)
		: RenderObject(updatable, conStruct)
		, m_mesh_structure(conStruct.mesh_structure, std::bind(&Mesh::NotifyMeshStructure, this))
	{
		for (uint32_t i = 0u; i < sm_mat_capacity; i++)
			m_materials[i].SetNotifyFunction(std::bind(&Mesh::NotifyMaterial, this));

		for (uint32_t i = 0u; i < GetMaterialCapacity(); i++)
		{
			SetMaterial(conStruct.material[i], i);
		}
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
		if (material_index >= GetMaterialCapacity()) return;
		if (!material) return;

		m_materials[material_index] = material;
		GetStateRegister().RequestUpdate();
	}

	const Handle<MeshStructure>& Mesh::GetStructure() const
	{
		return static_cast<const Handle<MeshStructure>&>(m_mesh_structure);
	}
	const Handle<Material>& Mesh::GetMaterial(uint32_t material_index) const
	{
		return m_materials[std::min(material_index, GetMaterialCapacity() - 1u)];
	}
	const Handle<Material> Mesh::GetMaterial(const std::string& material_name) const
	{
		const auto& material = std::find_if(m_materials.begin(), m_materials.end(), 
			[&material_name](auto& material) -> bool {
				return (material) ? (material->GetName() == material_name) : false;
			});

		if (material == m_materials.end()) return {};
		return *material;
	}
	uint32_t Mesh::GetMaterialIdx(const std::string& material_name) const
	{
		const auto& material = std::find_if(m_materials.begin(), m_materials.end(),
			[&material_name](auto& material) -> bool {
				return (material) ? (material->GetName() == material_name) : false;
			});

		return uint32_t(material - m_materials.begin());
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
			vN[i].RotateZYX(-GetTransformation().GetRotation());
		}

		// expand planes by each farthest (for plane direction) vertex
		auto& vertices = m_mesh_structure->GetVertices();
		for (unsigned int i = 0u; i < vertices.GetCount(); ++i)
		{
			Math::vec3f V = vertices[i];
			V *= GetTransformation().GetScale();

			for (int j = 0; j < 6; j++)
			{
				if (Math::vec3f::DotProduct(V - P[j], vN[j]) > 0.0f)
					P[j] = V;
			}
		}

		// rotate planes back
		for (int i = 0; i < 6; i++)
		{
			P[i].RotateXYZ(GetTransformation().GetRotation());
		}

		// set bounding box extents
		m_bounding_box.min.x = P[2].x;
		m_bounding_box.min.y = P[1].y;
		m_bounding_box.min.z = P[4].z;
		m_bounding_box.max.x = P[3].x;
		m_bounding_box.max.y = P[0].y;
		m_bounding_box.max.z = P[5].z;

		// transpose extents by object position
		m_bounding_box.min += GetTransformation().GetPosition();
		m_bounding_box.max += GetTransformation().GetPosition();
	}
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
}