#ifndef INSTANCE_H
#define INSTANCE_H

#include "world_object.hpp"
#include "material.hpp"
#include "mesh.hpp"
#include "groupable.hpp"

namespace RayZath::Engine
{
	class Instance;
	template<> struct ConStruct<Instance>;

	class Instance : public WorldObject, public Groupable
	{
	private:
		static constexpr uint32_t sm_mat_capacity = 64u;

		Transformation m_transformation;
		BoundingBox m_bounding_box;

		Observer<Mesh> m_mesh;
		std::array<Observer<Material>, sm_mat_capacity> m_materials;


	public:
		Instance(const Instance&) = delete;
		Instance(Instance&&) = delete;
		Instance(
			Updatable* updatable,
			const ConStruct<Instance>& conStruct);


	public:
		Instance& operator=(const Instance&) = delete;
		Instance& operator=(Instance&&) = delete;


	public:
		void position(const Math::vec3f& position);
		void rotation(const Math::vec3f& rotation);
		void scale(const Math::vec3f& scale);
		void lookAtPoint(const Math::vec3f& point, const Math::angle_radf& angle = 0.0f);
		void lookInDirection(const Math::vec3f& direction, const Math::angle_radf& angle = 0.0f);

		const Transformation& transformation() const;
		const BoundingBox& boundingBox() const;

		void mesh(const Handle<Mesh>& mesh);
		const Handle<Mesh>& mesh() const;

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
		void notifyMesh();
		void notifyMaterial();
	private:
		void calculateBoundingBox();
	};


	template<> struct ConStruct<Instance> : public ConStruct<WorldObject>
	{
		Math::vec3f position;
		Math::vec3f rotation;
		Math::vec3f scale;

		Handle<Mesh> mesh;
		Handle<Material> material[Instance::materialCapacity()];

		ConStruct(
			const std::string& name = "name",
			const Math::vec3f& position = Math::vec3f(0.0f, 0.0f, 0.0f),
			const Math::vec3f& rotation = Math::vec3f(0.0f, 0.0f, 0.0f),
			const Math::vec3f& scale = Math::vec3f(1.0f, 1.0f, 1.0f),
			const Handle<Mesh>& mesh = {},
			const Handle<Material>& mat = {})
			: ConStruct<WorldObject>(name)
			, position(position)
			, rotation(rotation)
			, scale(scale)
			, mesh(mesh)
		{
			material[0] = mat;
		}
		ConStruct(const Handle<Instance>& instance)
		{
			if (!instance) return;

			name = instance->name();
						
			position = instance->transformation().position();
			rotation = instance->transformation().rotation();
			scale = instance->transformation().scale();

			mesh = instance->mesh();
			for (uint32_t i = 0; i < uint32_t(sizeof(material) / sizeof(material[0])); i++)
				material[i] = instance->material(i);
		}
	};
}

#endif // !MESH_H