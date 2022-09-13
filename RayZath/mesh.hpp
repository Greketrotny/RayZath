#ifndef MESH_H
#define MESH_H

#include "world_object.hpp"
#include "material.hpp"
#include "mesh_structure.hpp"
#include "groupable.hpp"

namespace RayZath::Engine
{
	class Mesh;
	template<> struct ConStruct<Mesh>;

	class Mesh : public WorldObject, public Groupable
	{
	private:
		static constexpr uint32_t sm_mat_capacity = 64u;

		Transformation m_transformation;
		BoundingBox m_bounding_box;

		Observer<MeshStructure> m_mesh_structure;
		std::array<Observer<Material>, sm_mat_capacity> m_materials;


	public:
		Mesh(const Mesh&) = delete;
		Mesh(Mesh&&) = delete;
		Mesh(
			Updatable* updatable,
			const ConStruct<Mesh>& conStruct);


	public:
		Mesh& operator=(const Mesh&) = delete;
		Mesh& operator=(Mesh&&) = delete;


	public:
		void position(const Math::vec3f& position);
		void rotation(const Math::vec3f& rotation);
		void scale(const Math::vec3f& scale);
		void lookAtPoint(const Math::vec3f& point, const Math::angle_radf& angle = 0.0f);
		void lookInDirection(const Math::vec3f& direction, const Math::angle_radf& angle = 0.0f);

		const Transformation& transformation() const;
		const BoundingBox& boundingBox() const;

		void meshStructure(const Handle<MeshStructure>& mesh_structure);
		const Handle<MeshStructure>& meshStructure() const;

		void setMaterial(
			const Handle<Material>& material,
			const uint32_t& material_index);

		const Handle<Material>& material(uint32_t material_index) const;
		const Handle<Material> material(const std::string& material_name) const;
		uint32_t materialIdx(const std::string& material_name) const;
		static constexpr uint32_t materialCapacity()
		{
			return sm_mat_capacity;
		}
		void update() override;
		void notifyMeshStructure();
		void notifyMaterial();
	private:
		void calculateBoundingBox();
	};


	template<> struct ConStruct<Mesh> : public ConStruct<WorldObject>
	{
		Math::vec3f position;
		Math::vec3f rotation;
		Math::vec3f scale;

		Handle<MeshStructure> mesh_structure;
		Handle<Material> material[Mesh::materialCapacity()];

		ConStruct(
			const std::string& name = "name",
			const Math::vec3f& position = Math::vec3f(0.0f, 0.0f, 0.0f),
			const Math::vec3f& rotation = Math::vec3f(0.0f, 0.0f, 0.0f),
			const Math::vec3f& scale = Math::vec3f(1.0f, 1.0f, 1.0f),
			const Handle<MeshStructure>& mesh_structure = {},
			const Handle<Material>& mat = {})
			: ConStruct<WorldObject>(name)
			, position(position)
			, rotation(rotation)
			, scale(scale)
			, mesh_structure(mesh_structure)
		{
			material[0] = mat;
		}
		ConStruct(const Handle<Mesh>& mesh)
		{
			if (!mesh) return;

			name = mesh->name();
						
			position = mesh->transformation().position();
			rotation = mesh->transformation().rotation();
			scale = mesh->transformation().scale();

			mesh_structure = mesh->meshStructure();
			for (uint32_t i = 0; i < uint32_t(sizeof(material) / sizeof(material[0])); i++)
				material[i] = mesh->material(i);
		}
	};
}

#endif // !MESH_H