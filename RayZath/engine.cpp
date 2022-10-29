#include "engine.hpp"

namespace RayZath::CPU
{
	void Engine::renderWorld(
		RayZath::Engine::World& hWorld,
		const RayZath::Engine::RenderConfig& render_config,
		[[maybe_unused]] const bool block,
		[[maybe_unused]] const bool sync)
	{
		auto& cameras = hWorld.container<RayZath::Engine::World::ObjectType::Camera>();
		if (cameras.count() == 0) return;

		auto& camera = cameras[0];
		auto& image = camera->imageBuffer();
		for (uint32_t i = 100 + camera->position().x; i < 200; i++)
		{
			for (uint32_t j = 100; j < 200; j++)
			{
				if (i < image.GetHeight() && j < image.GetWidth())
					image.Value(j, i) = Graphics::Color::Palette::Red;
			}
		}
	}
}
