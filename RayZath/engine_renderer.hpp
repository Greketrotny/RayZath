#ifndef ENGINE_RENDERER_HPP
#define ENGINE_RENDERER_HPP

#include <thread>
#include <condition_variable>

namespace RayZath::CPU
{
	class EngineCore;

	class Renderer
	{
	private:
		std::reference_wrapper<EngineCore> mr_engine_core;

		std::thread m_render_thread;
		std::condition_variable m_render_cv;
		std::mutex m_render_mtx;
		bool m_terminate_render_thread = false;

	public:
		Renderer(std::reference_wrapper<EngineCore> engine_core);

		void launch();
		void terminate();

		void notify();

	private:
		void renderFunction();
	};
}

#endif
