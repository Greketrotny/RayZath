#include "headless.hpp"

#include "rayzath.hpp"

#include <chrono>
#include <fstream>
#include <iostream>
#include <format>

#include "text_utils.h"
#include "lib/Json/json.hpp"

namespace RayZath::Headless
{
	Headless& Headless::instance()
	{
		static Headless instance;
		return instance;
	}

	int Headless::run(
		std::filesystem::path scene_path,
		std::filesystem::path report_path,
		std::filesystem::path config_path)
	{
		auto tasks = prepareTasks(scene_path);
		std::vector<std::chrono::duration<float>> durations;
		for (const auto& task : tasks)
			durations.push_back(executeTask(task));

		// Generate report
		/*{
			if (report_path.empty())
			{
				std::cout << "No report path specified.";
				report_path = std::filesystem::current_path();
			}

			const auto start = std::chrono::steady_clock::now();
			std::cout << "Generating report in " << report_path << "\n";

			auto& cameras = world.container<RayZath::Engine::ObjectType::Camera>();
			for (uint32_t camera_id = 0; camera_id < cameras.count(); camera_id++)
			{
				auto& camera = cameras[camera_id];
				std::cout << "Saving rendered image of \"" << camera->name() << "\"";
				world.saver().saveMap<RayZath::Engine::ObjectType::Texture>(
					camera->imageBuffer(), report_path, camera->name());
			}

			const auto stop = std::chrono::steady_clock::now();
			std::cout << std::format(
				"\nGenerated report in: {:.3f}s\n",
				std::chrono::duration<float>(stop - start).count());
		}*/

		return 0;
	}

	std::vector<RenderTask> Headless::prepareTasks(const std::filesystem::path& benchmark_file)
	{
		using namespace std::string_literals;
		using json_t = nlohmann::json;
		try
		{
			RZAssert(
				benchmark_file.has_filename() &&
				std::filesystem::exists(benchmark_file),
				"Invalid path");

			// open file
			std::cout << "Reading config file: " << benchmark_file << std::endl;
			const auto start = std::chrono::steady_clock::now();
			std::ifstream file(benchmark_file, std::ios_base::in);
			RZAssert(file.is_open(), "Failed to open the file.");
			const auto json{json_t::parse(file, nullptr, true, true)};

			// prepare tasks
			static constexpr auto benchmarks_key = "benchmarks";
			RZAssert(json.contains(benchmarks_key), "File must contain "s + benchmarks_key + " key.");
			const auto& benchmarks_json = json[benchmarks_key];

			auto createTask = [&](const json_t& entry_json) {
				RenderTask task{};
				static constexpr auto scene_path_key = "scene path";
				RZAssert(entry_json.is_object(), "Benchmark entry must be an object.");
				RZAssert(entry_json.contains(scene_path_key), "Benchmark entry must contain a "s + scene_path_key + " key.");

				// load scene path
				const auto& scene_path_json = entry_json[scene_path_key];
				RZAssert(scene_path_json.is_string(), scene_path_key + " key must be a string"s);
				task.scene_path = static_cast<std::string>(scene_path_json);
				if (task.scene_path.is_relative())
					task.scene_path = benchmark_file.parent_path() / task.scene_path;

				// load rpp
				static constexpr auto rpp_key = "rpp";
				if (entry_json.contains(rpp_key))
					task.rpp = static_cast<uint32_t>(entry_json[rpp_key]);

				// load timeout
				static constexpr auto timeout_key = "timeout";
				if (entry_json.contains(timeout_key))
					task.timeout = static_cast<float>(entry_json[timeout_key]);

				return task;
			};

			std::vector<RenderTask> tasks;
			if (benchmarks_json.is_object())
			{
				tasks.push_back(createTask(benchmarks_json));
			}
			else if (benchmarks_json.is_array())
			{
				for (const auto& json_entry : benchmarks_json)
					tasks.push_back(createTask(json_entry));
			}
			else
				RZThrow(benchmarks_key + "'s value have to be either an array or an object."s);

			const auto stop = std::chrono::steady_clock::now();
			const auto duration = std::chrono::duration<float, std::milli>(stop - start);
			std::cout << std::format("Configured in: {:.3f}s\n\n", duration.count() / 1000.0f);

			return tasks;
		}
		catch (std::exception& e)
		{
			RZThrow("Failed to read file: " + benchmark_file.string() + ": " + e.what());
		}
	}
	std::chrono::duration<float> Headless::executeTask(const RenderTask& task)
	{
		auto& engine = Engine::Engine::instance();
		auto& world = engine.world();

		// Load scene(s)
		{
			std::cout << "Loading " << task.scene_path.filename() << std::endl;
			const auto start = std::chrono::steady_clock::now();
			world.loader().loadScene(task.scene_path);
			const auto stop = std::chrono::steady_clock::now();
			std::cout << std::format(
				"Loaded in: {:.3f}s\n\n",
				std::chrono::duration<float, std::milli>(stop - start).count() / 1000.0f);
		}

		engine.renderEngine(task.engine);
		engine.renderConfig().tracing().maxDepth(task.max_depth);

		// Render
		{
			const auto start = std::chrono::steady_clock::now();
			std::cout << "Rendering... 0%";
			size_t last_message_length = 0;

			size_t total_traced_rays = 0;
			auto& cameras = world.container<RayZath::Engine::ObjectType::Camera>();

			static constexpr std::array stick_array{'|', '/', '-', '\\'};
			int stick_id = 0;
			for (uint32_t traced = 0; traced < task.rpp;)
			{
				if (task.rpp - traced < engine.renderConfig().tracing().rpp())
					engine.renderConfig().tracing().rpp(task.rpp - traced);

				const auto pass_start = std::chrono::steady_clock::now();
				render();
				const auto stop = std::chrono::steady_clock::now();
				const auto task_duration = std::chrono::duration<float>(stop - start);
				const auto pass_duration = std::chrono::duration<float>(stop - pass_start);

				traced += engine.renderConfig().tracing().rpp();

				size_t pass_sum = 0;
				for (uint32_t i = 0; i < cameras.count(); i++)
					pass_sum += cameras[i]->rayCount();
				const auto ray_count_diff = pass_sum - total_traced_rays;
				total_traced_rays += ray_count_diff;

				const char stick = stick_array[stick_id];
				stick_id = (stick_id + 1) % stick_array.size();


				auto message = std::format(
					"\r{} Rendering... {}/{} +{} [rpp] ({:.2f}%) | {} rps | {:.3f}s (timeout: {:.3f}s)",
					stick,
					traced, task.rpp,
					engine.renderConfig().tracing().rpp(),
					(traced / float(task.rpp) * 100.0f),

					Utils::scientificWithPrefix(size_t(ray_count_diff / pass_duration.count())),

					task_duration.count(), task.timeout);
				std::cout << "\r" << std::string(last_message_length, ' ');
				last_message_length = message.length();
				std::cout << "\r" << message;

				if (task_duration.count() >= task.timeout)
					break;
			}

			const auto stop = std::chrono::steady_clock::now();
			const auto duration = std::chrono::duration<float>(stop - start);
			std::cout << std::format(
				"\nRendered in: {:.3f}s\n\n",
				duration.count());
			return duration;
		}
	}
	void Headless::render()
	{
		auto& engine = RayZath::Engine::Engine::instance();

		// call rendering engine
		const auto start = std::chrono::steady_clock::now();
		engine.renderWorld(true, false);
		const auto stop = std::chrono::steady_clock::now();
		const auto duration = std::chrono::duration<float>(stop - start).count();

		// balance the load
		const float relative_error = (duration - m_load_time) / m_load_time;
		if (std::abs(relative_error) > 0.05f)
		{
			const float duration_ratio = std::powf(m_load_time / duration, 0.5f);
			const float new_rpp = m_floaty_rpp * duration_ratio;
			m_floaty_rpp = (m_floaty_rpp + new_rpp) * 0.5f;
			engine.renderConfig().tracing().rpp(std::clamp<uint32_t>(uint32_t(m_floaty_rpp), 1, 1024));
		}
	}
}
