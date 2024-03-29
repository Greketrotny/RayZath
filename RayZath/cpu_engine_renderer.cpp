#include "cpu_engine_renderer.hpp"
#include "cpu_engine_core.hpp"
#include "cpu_engine_kernel.hpp"

#include "rayzath.hpp"

#include <iostream>
#include <random>
#include <numbers>
#include <utility>

namespace RayZath::Engine::CPU
{
	CameraContext::CameraContext(Math::vec2ui32 resolution)
	{
		resize(resolution);
	}
	void CameraContext::resize(const Math::vec2ui32 resolution)
	{
		if (resolution.x == m_image.GetWidth() && resolution.y == m_image.GetHeight())
			return;

		m_image.Resize(resolution.x, resolution.y);
		m_path_depth.Resize(resolution.x, resolution.y);

		m_ray_origin.Resize(resolution.x, resolution.y);
		m_ray_direction.Resize(resolution.x, resolution.y);
		m_ray_material.Resize(resolution.x, resolution.y);
		m_ray_color.Resize(resolution.x, resolution.y);
	}
	void CameraContext::reset(const Math::vec2ui32 resolution)
	{
		resize(resolution);		
		m_image.Clear(Graphics::ColorF(0.0f));

		m_update_flag = true;
		m_traced_rays = 0;
	}
	void CameraContext::setRay(const Math::vec2ui32 pixel, const SceneRay& ray)	
	{
		m_ray_origin.Value(pixel.x, pixel.y) = ray.origin;
		m_ray_direction.Value(pixel.x, pixel.y) = ray.direction;
		m_ray_material.Value(pixel.x, pixel.y) = ray.material;
		m_ray_color.Value(pixel.x, pixel.y) = ray.color;
	}
	SceneRay CameraContext::getRay(const Math::vec2ui32 pixel)
	{
		return SceneRay(
			m_ray_origin.Value(pixel.x, pixel.y),
			m_ray_direction.Value(pixel.x, pixel.y),
			m_ray_material.Value(pixel.x, pixel.y),
			m_ray_color.Value(pixel.x, pixel.y));
	}

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
		auto& cameras = world.container<RayZath::Engine::ObjectType::Camera>();
		for (uint32_t camera_id = 0; camera_id < cameras.count(); camera_id++)
		{
			const auto& camera = cameras[camera_id];
			if (!camera) continue;

			if (!m_contexts.count(camera))
				m_contexts.emplace(
					std::piecewise_construct,
					std::forward_as_tuple(camera),
					std::forward_as_tuple(Math::vec2ui32{camera->width(), camera->height()}));
			if (camera->stateRegister().RequiresUpdate() || world.stateRegister().RequiresUpdate())
			{
				camera->update();
				m_contexts[camera].reset(camera->resolution());
			}
			else
			{
				m_contexts[camera].resize(camera->resolution());
			}			
		}
		m_time_table.update("update cameras");

		m_kernel.setWorld(world);
		world.update();

		const size_t thread_launch_num = m_worker_threads.size();
		m_curr_workers = thread_launch_num;
		for (size_t i = 0; i < thread_launch_num; i++)
			m_worker_threads[i].second = true;

		m_workers_cv.notify_all();
		std::unique_lock lock{m_renderer_mtx};
		m_renderer_cv.wait(lock, [this]() { return m_curr_workers == 0; });

		m_time_table.update("wait for render");

		for (auto& [camera, context] : m_contexts)
			context.m_block_id = 0;

		m_time_table.update("result tranfer");
		m_time_table.updateCycle("full cycle");
	}

	void Renderer::workerFunction(const size_t worker_id)
	{
		std::random_device rd;
		std::mt19937 gen(rd() + uint32_t(worker_id));
		std::uniform_real_distribution<float> distr(0.0f, 1.0f);

		while (!m_terminate_worker_thread)
		{
			{
				std::unique_lock lock{m_workers_mtx};
				m_workers_cv.wait(
					lock,
					[&worker_id, this]() {
						return m_worker_threads[worker_id].second.load(); });
				m_worker_threads[worker_id].second = false;
				if (m_terminate_worker_thread)
					continue;
			}

			for (auto& [camera, context] : m_contexts)
			{
				if (!camera) continue;
				if (!camera->enabled()) continue;
				renderCameraView(*camera, context, RNG{Math::vec2f32{distr(gen), distr(gen)}, distr(gen)});
			}

			// last thread
			if (--m_curr_workers == 0)
			{
				for (auto& [camera, context] : m_contexts)
				{
					if (!camera) continue;
					context.m_traced_rays += uint64_t(camera->resolution().x) * camera->resolution().y;
					camera->rayCount(context.m_traced_rays);

					m_kernel.rayCast(*camera);
				}

				for (auto& context : m_contexts)
					context.second.m_update_flag = false;
				m_renderer_cv.notify_all();
			}
		}
	}

	void Renderer::renderCameraView(Camera& camera, CameraContext& context, RNG rng)
	{
		static constexpr Math::vec2ui32 block_size{128, 128};
		const auto x_blocks = ((camera.width() - 1) / block_size.x) + 1;
		const auto y_blocks = ((camera.height() - 1) / block_size.y) + 1;
		const auto block_count = x_blocks * y_blocks;

		World* world = mr_engine_core.get().mp_world;
		if (!world) return;

		const auto aperture_area = camera.aperture() * camera.aperture() * std::numbers::pi_v<float>;
		const auto exposure_time{camera.exposureTime()};

		const auto& render_config = RayZath::Engine::Engine::instance().renderConfig();

		if (context.m_update_flag)
		{
			for (uint32_t my_block_id = context.m_block_id++; 
				my_block_id < block_count; 
				my_block_id = context.m_block_id++)
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
						auto color{m_kernel.renderFirstPass(
							camera, 
							context, 
							Math::vec2ui32{x, y}, 
							rng,
							render_config)};
						color /= color.alpha == 0.0f ? 1.0f : color.alpha;

						color *= aperture_area;
						color *= exposure_time;
						color *= 1.0e5f;	// camera matrix sensitivity.
						color = color / (color + Graphics::ColorF(1.0f));

						camera.imageBuffer().Value(x, y) = Graphics::Color(
							uint8_t(color.red * 255.0f),
							uint8_t(color.green * 255.0f),
							uint8_t(color.blue * 255.0f),
							255);
					}
				}
			}
		}
		else
		{
			for (uint32_t my_block_id = context.m_block_id++; 
				my_block_id < block_count; 
				my_block_id = context.m_block_id++)
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
						auto color{m_kernel.renderCumulativePass(
							camera, 
							context, 
							Math::vec2ui32{x, y}, 
							rng,
							render_config)};
						color /= color.alpha == 0.0f ? 1.0f : color.alpha;

						color *= aperture_area;
						color *= exposure_time;
						color *= 1.0e5f;	// camera matrix sensitivity.
						color = (color / (color + Graphics::ColorF(1.0f)));

						camera.imageBuffer().Value(x, y) = Graphics::Color(
							uint8_t(color.red * 255.0f),
							uint8_t(color.green * 255.0f),
							uint8_t(color.blue * 255.0f),
							255);
					}
				}
			}
		}		
	}
}
