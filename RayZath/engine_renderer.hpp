#ifndef ENGINE_RENDERER_HPP
#define ENGINE_RENDERER_HPP

#include <vector>
#include <thread>
#include <condition_variable>

namespace RayZath::CPU
{
	class EngineCore;

	class Renderer
	{
	private:
		std::reference_wrapper<EngineCore> mr_engine_core;

		// --- renderer ---
		std::mutex m_renderer_mtx;
		std::condition_variable m_renderer_cv;

		// --- workers ---
		std::atomic<size_t> m_curr_workers = 0;
		bool m_terminate_worker_thread = false;
		std::mutex m_workers_mtx;
		std::condition_variable m_workers_cv;
		std::vector<std::pair<std::thread, std::atomic<bool>>> m_worker_threads;

	public:
		Renderer(std::reference_wrapper<EngineCore> engine_core);
		~Renderer();

		void launch();
		void terminate();

		void render();

	private:
		void workerFunction(const uint32_t worker_id);
	};
}

#endif
