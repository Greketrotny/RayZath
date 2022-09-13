#include "mesh.hpp"
#include "group.hpp"

namespace RayZath::Engine
{
	Mesh::Mesh(
		Updatable* updatable,
		const ConStruct<Mesh>& construct)
		: WorldObject(updatable, construct)
		, m_transformation(construct.position, construct.rotation, construct.scale)
		, m_mesh_structure(construct.mesh_structure, std::bind(&Mesh::notifyMeshStructure, this))
	{
		for (uint32_t i = 0u; i < sm_mat_capacity; i++)
			m_materials[i].setNotifyFunction(std::bind(&Mesh::notifyMaterial, this));

		for (uint32_t i = 0u; i < materialCapacity(); i++)
			setMaterial(construct.material[i], i);
	}


	void Mesh::position(const Math::vec3f& position)
	{
		m_transformation.position(position);
		stateRegister().RequestUpdate();
	}
	void Mesh::rotation(const Math::vec3f& rotation)
	{
		m_transformation.rotation(rotation);
		stateRegister().RequestUpdate();
	}
	void Mesh::scale(const Math::vec3f& scale)
	{
		m_transformation.scale(scale);
		stateRegister().RequestUpdate();
	}
	void Mesh::lookAtPoint(const Math::vec3f& point, const Math::angle_radf& angle)
	{
		m_transformation.lookAtPoint(point, angle);
		stateRegister().RequestUpdate();
	}
	void Mesh::lookInDirection(const Math::vec3f& direction, const Math::angle_radf& angle)
	{
		m_transformation.lookInDirection(direction, angle);
		stateRegister().RequestUpdate();
	}

	const Transformation& Mesh::transformation() const
	{
		return m_transformation;
	}
	const BoundingBox& Mesh::boundingBox() const
	{
		return m_bounding_box;
	}

	void Mesh::meshStructure(const Handle<MeshStructure>& mesh_structure)
	{
		m_mesh_structure = mesh_structure;
		stateRegister().RequestUpdate();
	}
	void  Mesh::setMaterial(
		const Handle<Material>& material,
		const uint32_t& material_index)
	{
		if (material_index >= materialCapacity()) return;
		if (!material) return;

		m_materials[material_index] = material;
		stateRegister().RequestUpdate();
	}

	const Handle<MeshStructure>& Mesh::meshStructure() const
	{
		return static_cast<const Handle<MeshStructure>&>(m_mesh_structure);
	}
	const Handle<Material>& Mesh::material(uint32_t material_index) const
	{
		return m_materials[std::min(material_index, materialCapacity() - 1u)];
	}
	const Handle<Material> Mesh::material(const std::string& material_name) const
	{
		const auto& material = std::find_if(m_materials.begin(), m_materials.end(),
			[&material_name](auto& material) -> bool {
				return (material) ? (material->name() == material_name) : false;
			});

		if (material == m_materials.end()) return {};
		return *material;
	}
	uint32_t Mesh::materialIdx(const std::string& material_name) const
	{
		const auto& material = std::find_if(m_materials.begin(), m_materials.end(),
			[&material_name](auto& material) -> bool {
				return (material) ? (material->name() == material_name) : false;
			});

		return uint32_t(material - m_materials.begin());
	}

	void Mesh::update()
	{
		if (!stateRegister().RequiresUpdate()) return;

		calculateBoundingBox();

		stateRegister().update();
	}
	void Mesh::notifyMeshStructure()
	{
		stateRegister().MakeModified();
	}
	void Mesh::notifyMaterial()
	{
		stateRegister().MakeModified();
	}

	void Mesh::calculateBoundingBox()
	{
		m_bounding_box = BoundingBox();

		if (!m_mesh_structure) return;
		auto& vertices = m_mesh_structure->vertices();
		if (vertices.count() == 0) return;

		m_transformation_in_group = m_transformation;
		Handle<Group> subgroup = group();
		while (subgroup)
		{
			m_transformation_in_group *= subgroup->transformation();
			subgroup = subgroup->group();
		}

		Math::vec3f x_axis = m_transformation_in_group.coordSystem().xAxis();
		Math::vec3f y_axis = m_transformation_in_group.coordSystem().yAxis();
		Math::vec3f z_axis = m_transformation_in_group.coordSystem().zAxis();

		// expand planes by each farthest (for plane direction) vertex
		auto first_vertex = vertices[0];
		first_vertex *= m_transformation_in_group.scale();
		first_vertex = x_axis * first_vertex.x + y_axis * first_vertex.y + z_axis * first_vertex.z;

		m_bounding_box = BoundingBox(first_vertex, first_vertex);
		for (uint32_t i = 1; i < vertices.count(); ++i)
		{
			Math::vec3f vertex = vertices[i];
			vertex *= m_transformation_in_group.scale();
			vertex = x_axis * vertex.x + y_axis * vertex.y + z_axis * vertex.z;

			m_bounding_box.extendBy(vertex);
		}

		// transpose extents by object position
		m_bounding_box.min += m_transformation_in_group.position();
		m_bounding_box.max += m_transformation_in_group.position();
	}
}
