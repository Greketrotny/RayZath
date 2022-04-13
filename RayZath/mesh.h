#ifndef MESH_H
#define MESH_H

#include "world_object.h"
#include "material.h"
#include "mesh_structure.h"

namespace RayZath::Engine
{
	class Mesh;
	template<> struct ConStruct<Mesh>;

	class Mesh : public WorldObject
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
		void SetPosition(const Math::vec3f& position);
		void SetRotation(const Math::vec3f& rotation);
		void SetScale(const Math::vec3f& scale);
		void LookAtPoint(const Math::vec3f& point, const Math::angle_radf& angle = 0.0f);
		void LookInDirection(const Math::vec3f& direction, const Math::angle_radf& angle = 0.0f);

		const Transformation& GetTransformation() const;
		const BoundingBox& GetBoundingBox() const;

		void SetMeshStructure(const Handle<MeshStructure>& mesh_structure);
		const Handle<MeshStructure>& GetStructure() const;

		void SetMaterial(
			const Handle<Material>& material,
			const uint32_t& material_index);

		const Handle<Material>& GetMaterial(uint32_t material_index) const;
		const Handle<Material> GetMaterial(const std::string& material_name) const;
		uint32_t GetMaterialIdx(const std::string& material_name) const;
		static constexpr uint32_t GetMaterialCapacity()
		{
			return sm_mat_capacity;
		}
	public:
		void Update() override;
		void NotifyMeshStructure();
		void NotifyMaterial();
	private:
		void CalculateBoundingBox();
	};


	template<> struct ConStruct<Mesh> : public ConStruct<WorldObject>
	{
		Math::vec3f position;
		Math::vec3f rotation;
		Math::vec3f scale;

		Handle<MeshStructure> mesh_structure;
		Handle<Material> material[Mesh::GetMaterialCapacity()];

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
	};
}

#endif // !MESH_H