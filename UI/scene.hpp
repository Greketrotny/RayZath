#pragma once

#include "rayzath.hpp"

namespace RZ = RayZath::Engine;

namespace RayZath::UI
{
	class Scene
	{
	private:
	public:
		RZ::Engine& mr_engine;
		RZ::World& mr_world;

		Scene();

		void init();
		void createDefaultScene();

		void render();
		void update(const float elapsed_time);
	};
}