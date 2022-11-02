#include "engine_renderer.hpp"

#include "engine_core.hpp"

#include <iostream>

namespace RayZath::CPU
{
	Renderer::Renderer(std::reference_wrapper<EngineCore> engine_core)
		: mr_engine_core(std::move(engine_core))
		, m_worker_threads(std::thread::hardware_concurrency())
	{
		launch();
	}
	Renderer::~Renderer()
	{
		terminate();
	}

	void Renderer::launch()
	{
		m_terminate_worker_thread = false;
		for (size_t worker_id = 0; worker_id < m_worker_threads.size(); worker_id++)
		{
			auto& [thread, run_flag] = m_worker_threads[worker_id];
			if (!thread.joinable())
			{
				run_flag = false;
				thread = std::thread(
					&Renderer::workerFunction,
					this,
					worker_id);
			}
		}
	}
	void Renderer::terminate()
	{
		m_terminate_worker_thread = true;
		for (auto& [thread, run_flag] : m_worker_threads)
		{
			run_flag = true;
			m_workers_cv.notify_all();
			thread.join();
		}
	}
	void Renderer::render()
	{
		for (auto& [worker_thread, run_flag] : m_worker_threads)
			run_flag = true;
		m_curr_workers = m_worker_threads.size();
		m_workers_cv.notify_all();
		std::unique_lock lock{m_renderer_mtx};
		m_renderer_cv.wait(lock, [this]() { return m_curr_workers == 0; });
	}

	void Renderer::workerFunction(const uint32_t worker_id)
	{
		while (!m_terminate_worker_thread)
		{
			{
				std::unique_lock lock{m_workers_mtx};
				m_workers_cv.wait(
					lock, 
					[&worker_id, this]() { 
						return m_worker_threads[worker_id].second && m_curr_workers != 0; });
				m_worker_threads[worker_id].second = false;
				if (m_terminate_worker_thread)
					continue;
			}

			auto& world = *mr_engine_core.get().mp_world;
			auto& cameras = world.container<RayZath::Engine::World::ObjectType::Camera>();
			for (uint32_t camera_id = 0; camera_id < cameras.count(); camera_id++)
			{
				auto& camera = cameras[camera_id];
				if (!camera) continue;

				const auto y_begin = camera->height() * worker_id / uint32_t(m_worker_threads.size());
				const auto y_end = camera->height() * (worker_id + 1) / uint32_t(m_worker_threads.size());

				for (uint32_t y = y_begin; y < y_end; y++)
				{
					for (uint32_t x = 0; x < camera->width(); x++)
						camera->imageBuffer().Value(x, y) = Graphics::Color(y_begin * 255 / camera->height());
				}
			}

			if (--m_curr_workers == 0)
			{
				m_renderer_cv.notify_all();
			}
		}
	}
}
