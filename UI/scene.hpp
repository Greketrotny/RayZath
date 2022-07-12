#pragma once

#include "rayzath.h"

namespace RZ = RayZath::Engine;

namespace RayZath::UI
{
	enum class CommonMesh
	{
		Plane,
		Sphere,
		Cone,
		Cylinder,
		Torus
	};
	template <CommonMesh T>
	struct CommonMeshParameters;
	template<>
	struct CommonMeshParameters<CommonMesh::Plane>
	{
		uint32_t sides = 4;
		float width = 1.0f, height = 1.0f;

	public:
		CommonMeshParameters(const uint32_t sides = 4, const float width = 1.0f, const float height = 1.0f)
			: sides(sides)
			, width(width)
			, height(height)
		{}
	};
	template<>
	struct CommonMeshParameters<CommonMesh::Cylinder>
	{
		uint32_t faces = 16;
		bool smooth_shading = true;

	public:
		CommonMeshParameters(const uint32_t faces = 16)
			: faces(faces)
		{}
	};

	class Scene
	{
	private:
	public:
		RZ::Engine& mr_engine;
		RZ::World& mr_world;

		std::vector<std::string> m_scene_files;
		std::string m_base_scene_path;

		Scene();

		void init();
		void loadScene(size_t scene_id = 0u);

		void render();
		void update(const float elapsed_time);

		template <CommonMesh T>
		RZ::Handle<RZ::MeshStructure> Create(const CommonMeshParameters<T>& parameters);
	};
}