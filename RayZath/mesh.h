#ifndef MESH_H
#define MESH_H

#include "render_object.h"
#include "mesh_structure.h"

namespace RayZath
{
	class Mesh;
	template<> struct ConStruct<Mesh>;

	class Mesh : public RenderObject
	{
		static constexpr uint32_t sm_mat_count = 64u;
	private:
		Observer<MeshStructure> m_mesh_structure;
		Observer<Material> m_materials[sm_mat_count];


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
		void SetMeshStructure(const Handle<MeshStructure>& mesh_structure);
		void SetMaterial(
			const Handle<Material>& material,
			const uint32_t& material_index);

		const Handle<MeshStructure>& GetStructure() const;
		const Handle<Material>& GetMaterial(const uint32_t& material_index) const;
		static constexpr uint32_t GetMaterialCount()
		{
			return sm_mat_count;
		}
	public:
		void Update() override;
		void NotifyMeshStructure();
		void NotifyMaterial();
	private:
		void CalculateBoundingBox();
	};


	template<> struct ConStruct<Mesh> : public ConStruct<RenderObject>
	{
		Handle<MeshStructure> mesh_structure;
		Handle<Material> material;

		ConStruct(
			const std::string& name = "name",
			const Math::vec3f& position = Math::vec3f(0.0f, 0.0f, 0.0f),
			const Math::vec3f& rotation = Math::vec3f(0.0f, 0.0f, 0.0f),
			const Math::vec3f& center = Math::vec3f(0.0f, 0.0f, 0.0f),
			const Math::vec3f& scale = Math::vec3f(1.0f, 1.0f, 1.0f),
			const Handle<MeshStructure>& mesh_structure = Handle<MeshStructure>(),
			const Handle<Material>& material = Handle<Material>())
			: ConStruct<RenderObject>(name, position, rotation, center, scale)
			, mesh_structure(mesh_structure)
			, material(material)
		{}
	};
}

#endif // !MESH_H