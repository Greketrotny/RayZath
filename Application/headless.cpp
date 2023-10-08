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
		std::filesystem::path report_dir,
		const bool save_images)
	{
		// check and prepare tasks
		const auto tasks{prepareTasks(scene_path)};

		// check report path
		if (report_dir.empty())
		{
			std::cout << "No report path specified.";
			report_dir = std::filesystem::current_path();
		}
		if (!std::filesystem::is_directory(report_dir))
			RZThrow("report_path must be a directory.");

		const auto now = std::chrono::system_clock::now();
		const auto dir_name = std::format("benchmark_{0:%Y%m%d}_{0:%H%M%S}", now);
		report_dir = report_dir / dir_name;
		std::filesystem::create_directories(report_dir);

		std::vector<TaskResult> results;
		for (const auto& task : tasks)
		{
			auto task_results = executeTask(task, report_dir, save_images);
			results.insert(
				results.end(),
				std::make_move_iterator(task_results.begin()), std::make_move_iterator(task_results.end()));
		}
		generateReport(report_dir, results);

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
			static constexpr auto task_key = "tasks";
			RZAssert(json.contains(task_key), "File must contain \""s + task_key + "\" key.");
			const auto& benchmarks_json = json[task_key];

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

				// engine 
				static constexpr auto engine_key = "engine";
				if (entry_json.contains(engine_key))
				{
					const auto& engine_json = entry_json[engine_key];

					auto load_engine = [&](const auto& engine_entry) {
						RZAssert(engine_entry.is_string(), "Specified engine must be a string.");
						const auto engine_str = std::string(engine_entry);
						for (const auto& [type, name] : Engine::Engine::engine_name)
							if (engine_str == name)
							{
								task.engine.push_back(type);
								return;
							}
						RZThrow("Unknown engine type \"" + engine_str + "\"");
					};
					if (engine_json.is_string())
					{
						load_engine(engine_json);
					}
					else if (engine_json.is_array())
					{
						for (const auto& entry : engine_json)
							load_engine(entry);
					}
					else
						RZThrow("Engine value must be either a string or an array.");
				}
				else
				{
					task.engine = {Engine::Engine::RenderEngine::CUDAGPU};
				}

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
				RZThrow(task_key + "'s value have to be either an array or an object."s);

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
	std::vector<TaskResult> Headless::executeTask(
		const RenderTask& task,
		const std::filesystem::path& report_dir,
		bool save_images)
	{
		// Load scene(s)
		{
			auto engine = Engine::Engine::mutableInstance();
			std::cout << "Loading " << task.scene_path.filename() << std::endl;
			const auto start = std::chrono::steady_clock::now();
			Engine::Loader(engine.)
			world.loader().loadScene(task.scene_path);
			const auto stop = std::chrono::steady_clock::now();
			std::cout << std::format(
				"Loaded in: {:.3f}s\n\n",
				std::chrono::duration<float, std::milli>(stop - start).count() / 1000.0f);
		}

		engine.renderConfig().tracing().maxDepth(task.max_depth);


		// Render
		std::vector<TaskResult> results;
		for (const auto& engine_type : task.engine)
		{
			engine.renderEngine(engine_type);
			TaskResult result(task);
			result.engine = engine_type;
			{
				std::cout << "Rendering...";
				m_floaty_rpp = 1.0f;
				std::size_t last_message_length = 0;
				static constexpr std::array stick_array{'|', '/', '-', '\\'};
				int stick_id = 0;

				auto& cameras = world.container<RayZath::Engine::ObjectType::Camera>();
				uint32_t traced = 0;
				if (task.rpp - traced < engine.renderConfig().tracing().rpp())
					engine.renderConfig().tracing().rpp(task.rpp - traced);
				render();

				const auto start = std::chrono::steady_clock::now();
				auto last_stop = start;
				for (traced = 0; traced < task.rpp;)
				{
					if (task.rpp - traced < engine.renderConfig().tracing().rpp())
						engine.renderConfig().tracing().rpp(task.rpp - traced);

					render();
					const auto stop = std::chrono::steady_clock::now();
					const auto task_duration = std::chrono::duration<float>(stop - start);
					const auto pass_duration = std::chrono::duration<float>(stop - last_stop);
					last_stop = stop;

					traced += engine.renderConfig().tracing().rpp();

					std::size_t pass_sum = 0;
					for (uint32_t i = 0; i < cameras.count(); i++)
						pass_sum += cameras[i]->rayCount();
					const auto ray_count_diff = pass_sum - result.total_traced_rays;
					result.total_traced_rays += ray_count_diff;

					const char stick = stick_array[stick_id];
					stick_id = (stick_id + 1) % stick_array.size();


					auto message = std::format(
						"\r{} Rendering... {}/{} +{} [rpp] ({:.2f}%) | {} rps | {:.3f}s (timeout: {:.3f}s)",
						stick,
						traced, task.rpp,
						engine.renderConfig().tracing().rpp(),
						(traced / float(task.rpp) * 100.0f),

						Utils::scientificWithPrefix(std::size_t(ray_count_diff / pass_duration.count())),

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
				result.duration = duration;

				if (save_images)
				{
					for (uint32_t camera_id = 0; camera_id < cameras.count(); camera_id++)
					{
						auto& camera = cameras[camera_id];
						std::cout << "Saving rendered image of \"" << camera->name() << "\"\n";
						const std::string image_file_name =
							std::format("{}_{}_{}_{}",
								result.scene_path.filename().string(),
								camera->name(),
								Utils::scientificWithPrefix(result.total_traced_rays),
								Engine::Engine::engine_name.at(result.engine));
						world.saver().saveMap<RayZath::Engine::ObjectType::Texture>(
							camera->imageBuffer(), report_dir, image_file_name);
					}
				}				
			}
			results.push_back(std::move(result));
		}

		return results;
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
	void Headless::generateReport(
		std::filesystem::path report_dir,
		const std::vector<TaskResult>& results)
	{
		const auto start = std::chrono::steady_clock::now();

		const auto report_file_path = report_dir / "report.txt";
		std::cout << "Generating report in " << report_file_path << "\n";
		std::ofstream report_file(report_file_path);

		for (std::size_t i = 0; i < results.size(); i++)
		{
			const auto& result = results[i];

			report_file << std::format(
				"Scene: {}\n",
				result.scene_path.filename().string());
			report_file << std::format("\tengine: {} | max depth: {}\n",
				Engine::Engine::engine_name.at(result.engine),
				result.max_depth);
			report_file << std::format("\tduration: {:.3f}s | traced {} rays ({} rps)",
				result.duration.count(),
				Utils::scientificWithPrefix(result.total_traced_rays),
				Utils::scientificWithPrefix(std::size_t(result.total_traced_rays / result.duration.count())));
			report_file << std::endl;

		}
		report_file.close();

		const auto stop = std::chrono::steady_clock::now();
		std::cout << std::format(
			"\nGenerated report in: {:.3f}s\n",
			std::chrono::duration<float>(stop - start).count());
	}
}
