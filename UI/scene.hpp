#pragma once

#include "rayzath.h"

namespace RZ = RayZath::Engine;

namespace RayZath::UI
{
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
	private:
		RZ::Handle<RZ::Mesh> CreateCube(
			RZ::World& world, RZ::ConStruct<RZ::Mesh> conStruct);
		/*void CreateTessellatedSphere(
			RZ::World* world,
			const RZ::ConStruct<RZ::Mesh>& conStruct,
			const uint32_t& resolution = 8u);*/
	};
}