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
		std::vector<Engine::Engine::RenderEngine> engine;
		uint8_t max_depth = 16;
	};
	struct TaskResult
	{
		std::filesystem::path scene_path{};
		Engine::Engine::RenderEngine engine{};
		std::chrono::duration<float> duration{};
		size_t total_traced_rays = 0;
		uint8_t max_depth = 16;

		TaskResult(const RenderTask& task)
			: scene_path(task.scene_path)
			, max_depth(task.max_depth)
		{}
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
			bool save_images);

		std::vector<RenderTask> prepareTasks(const std::filesystem::path& benchmark_file);
		std::vector<TaskResult> executeTask(
			const RenderTask& task, 
			const std::filesystem::path& report_dir,
			bool save_images);
		void render();
		void generateReport(
			std::filesystem::path report_dir, 
			const std::vector<TaskResult>& results);
	};
}

