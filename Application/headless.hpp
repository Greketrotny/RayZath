#pragma once

#include "rayzath.hpp"

#include <chrono>
#include <filesystem>

namespace RayZath::Headless
{
	struct RenderTask
	{
		std::filesystem::path scene_path;
		uint32_t rpp = 1000;
		float timeout = 60.0f;
		Engine::Engine::RenderEngine engine = Engine::Engine::RenderEngine::CUDAGPU;
		uint8_t max_depth = 16;
	};


	class Headless
	{
	private:
		float m_load_time = 0.1f, m_floaty_rpp = 1.0f;

	public:
		static Headless& instance();

		int run(
			std::filesystem::path scene_path,
			std::filesystem::path report_path,
			std::filesystem::path config_path);

		std::vector<RenderTask> prepareTasks(const std::filesystem::path& benchmark_file);
		std::chrono::duration<float> executeTask(const RenderTask& task);
		void render();
	};
}

