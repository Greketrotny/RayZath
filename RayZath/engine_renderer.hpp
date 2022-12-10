#ifndef ENGINE_RENDERER_HPP
#define ENGINE_RENDERER_HPP

#include "camera.hpp"
#include "roho.hpp"

#include "engine_parts.hpp"

#include <vector>
#include <thread>
#include <condition_variable>

namespace RayZath::Engine::CPU
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
				
		using acc_buffer_t = Graphics::Buffer2D<Graphics::ColorF>;
		std::unordered_map<Handle<Camera>, acc_buffer_t> m_accumulators;

		std::atomic<uint32_t> m_block_id = 0;

		TimeTable m_time_table;

	public:
		Renderer(std::reference_wrapper<EngineCore> engine_core);
		~Renderer();
		
		
		void launch();	
		void terminate();
		void render();

		const TimeTable& timeTable() const { return m_time_table; }

	private:
		void workerFunction(const uint32_t worker_id);

		void renderCameraView(
			const uint32_t worker_id, 
			Camera& camera, acc_buffer_t& acc_buffer);

		Graphics::ColorF render(const Camera& camera, const Math::vec2ui32 pixel);
	};
}

#endif
