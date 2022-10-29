#ifndef ENGINE_HPP
#define	ENGINE_HPP

#include "world.hpp"
#include "engine_parts.hpp"

namespace RayZath::CPU
{
	class Engine
	{
	public:
		void renderWorld(
			RayZath::Engine::World& hWorld,
			const RayZath::Engine::RenderConfig& render_config,
			const bool block = true,
			const bool sync = true);
	};
}

#endif
