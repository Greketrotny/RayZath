#ifndef ENGINE_HPP
#define	ENGINE_HPP

#include "world.hpp"
#include "engine_core.hpp"
#include "engine_parts.hpp"

namespace RayZath::Engine::CPU
{
	class Engine
	{
	private:
		EngineCore m_engine_core;
		std::string m_timing_string;

	public:
		void renderWorld(
			RayZath::Engine::World& hWorld,
			const RayZath::Engine::RenderConfig& render_config,
			const bool block = true,
			const bool sync = true);
		std::string timingsString();
	};
}

#endif
