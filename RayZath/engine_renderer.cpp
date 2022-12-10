#include "engine_renderer.hpp"

#include "engine_core.hpp"

#include <iostream>

namespace RayZath::Engine::CPU
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
		m_time_table.update("no render cycle");

		auto& world = *mr_engine_core.get().mp_world;
		auto& cameras = world.container<RayZath::Engine::World::ObjectType::Camera>();
		for (uint32_t camera_id = 0; camera_id < cameras.count(); camera_id++)
		{
			const auto& camera = cameras[camera_id];
			if (!camera) continue;

			if (!m_accumulators.count(camera))
				m_accumulators.insert({camera, acc_buffer_t{camera->width(), camera->height()}});
			if (camera->stateRegister().RequiresUpdate())
			{
				camera->update();
				m_accumulators[camera].Resize(camera->width(), camera->height(), Graphics::ColorF(0.0f));
			}
			else
			{
				m_accumulators[camera].Resize(camera->width(), camera->height());
			}
		}
		m_time_table.update("update cameras");

		for (auto& [worker_thread, run_flag] : m_worker_threads)
			run_flag = true;
		m_curr_workers = m_worker_threads.size();
		m_workers_cv.notify_all();
		std::unique_lock lock{m_renderer_mtx};
		m_renderer_cv.wait(lock, [this]() { return m_curr_workers == 0; });

		m_time_table.update("wait for render");

		m_block_id = 0;

		m_time_table.update("result tranfer");
		m_time_table.updateCycle("full cycle");
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
				auto& camera_ref = *camera;

				auto& image_buffer = m_accumulators[camera];
				renderCameraView(camera_ref, image_buffer);
			}

			if (--m_curr_workers == 0)
			{
				m_renderer_cv.notify_all();
			}
		}
	}

	void Renderer::renderCameraView(Camera& camera, acc_buffer_t& image_buffer)
	{
		static constexpr Math::vec2ui32 block_size{128, 128};
		const auto x_blocks = ((camera.width() - 1) / block_size.x) + 1;
		const auto y_blocks = ((camera.height() - 1) / block_size.y) + 1;
		const auto block_count = x_blocks * y_blocks;

		for (uint32_t my_block_id = m_block_id++; my_block_id < block_count; my_block_id = m_block_id++)
		{
			const auto block_y = my_block_id / x_blocks;
			const auto block_x = my_block_id % x_blocks;
			const Math::vec2ui32 top_left{block_x * block_size.x, block_y * block_size.y};
			Math::vec2ui32 bottom_right = top_left + block_size;
			bottom_right.x = std::min(bottom_right.x, camera.width());
			bottom_right.y = std::min(bottom_right.y, camera.height());

			for (uint32_t y = top_left.y; y != bottom_right.y; y++)
			{
				for (uint32_t x = top_left.x; x != bottom_right.x; x++)
				{
					auto color{image_buffer.Value(x, y) += render(camera, Math::vec2ui32{x, y})};
					color = color / color.alpha;
					color = (color / (color + Graphics::ColorF(1.0f)));

					camera.imageBuffer().Value(x, y) = Graphics::Color(
						uint8_t(color.red * 255.0f),
						uint8_t(color.green * 255.0f),
						uint8_t(color.blue * 255.0f),
						255);
					camera.depthBuffer().Value(x, y) = 10.0f;
				}
			}
		}
	}
	Graphics::ColorF CPU::Renderer::render(
		const Camera& camera, 
		const Math::vec2ui32 pixel)
	{
		SceneRay ray{};
		camera.generateRay(ray, pixel);

		return Graphics::ColorF(
			(ray.direction.x + 1.0f) / 2.0f, 
			(ray.direction.y + 1.0f) / 2.0f, 
			(ray.direction.z + 1.0f) / 2.0f, 
			1.0f);
	}
}
