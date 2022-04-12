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
		m_bounding_box = BoundingBox();

		if (!m_mesh_structure) return;
		auto& vertices = m_mesh_structure->GetVertices();
		if (vertices.GetCount() == 0) return;

		Math::vec3f x_axis(1.0f, 0.0f, 0.0f);
		Math::vec3f y_axis(0.0f, 1.0f, 0.0f);
		Math::vec3f z_axis(0.0f, 0.0f, 1.0f);
		x_axis.RotateXYZ(GetTransformation().GetRotation());
		y_axis.RotateXYZ(GetTransformation().GetRotation());
		z_axis.RotateXYZ(GetTransformation().GetRotation());

		// expand planes by each farthest (for plane direction) vertex
		auto first_vertex = vertices[0];
		first_vertex *= GetTransformation().GetScale();
		first_vertex = x_axis * first_vertex.x + y_axis * first_vertex.y + z_axis * first_vertex.z;

		m_bounding_box = BoundingBox(first_vertex, first_vertex);
		for (uint32_t i = 1; i < vertices.GetCount(); ++i)
		{
			Math::vec3f vertex = vertices[i];
			vertex *= GetTransformation().GetScale();
			vertex = x_axis * vertex.x + y_axis * vertex.y + z_axis * vertex.z;

			m_bounding_box.ExtendBy(vertex);
		}

		// transpose extents by object position
		m_bounding_box.min += GetTransformation().GetPosition();
		m_bounding_box.max += GetTransformation().GetPosition();
	}
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
}