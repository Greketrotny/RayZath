#include "engine_renderer.hpp"

#include "engine_core.hpp"

namespace RayZath::CPU
{
	Renderer::Renderer(std::reference_wrapper<EngineCore> engine_core)
		: mr_engine_core(std::move(engine_core))
	{}

	void Renderer::launch()
	{
		if (!m_render_thread.joinable())
		{
			m_terminate_render_thread = false;
			//resetExceptions();
			m_render_thread = std::thread(
				&Renderer::renderFunction,
				this);
		}
	}
	void Renderer::terminate()
	{
		if (m_render_thread.joinable())
		{
			m_terminate_render_thread = true;
			//mp_engine_core->fenceTrack().openAll();
			m_render_thread.join();
		}
	}

	void renderFunction()
	{

	}
}
