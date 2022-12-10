#ifndef ENGINE_CORE_HPP
#define ENGINE_CORE_HPP

#include "engine_renderer.hpp"
#include "engine_parts.hpp"
#include "world.hpp"

namespace RayZath::Engine::CPU
{
	class EngineCore
	{
	private:
		Renderer m_renderer;

		std::mutex m_mtx;

	public:
		RayZath::Engine::World* mp_world = nullptr;


	public:
		EngineCore();

		void renderWorld(
			RayZath::Engine::World& hWorld,
			const RayZath::Engine::RenderConfig& render_config,
			const bool block = true,
			const bool sync = true);
		const RayZath::Engine::TimeTable& timeTable() const;
	};
}

#endif
