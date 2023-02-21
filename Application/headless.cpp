#include "headless.hpp"

#include "rayzath.hpp"

#include <chrono>
#include <fstream>
#include <iostream>
#include <format>

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
		using namespace std::chrono_literals;

		auto& engine = RayZath::Engine::Engine::instance();
		auto& world = engine.world();

		// Load scene(s)
		{
			std::cout << "Loading " << scene_path.filename() << std::endl;
			const auto start = std::chrono::steady_clock::now();
			world.loader().loadScene(scene_path);
			const auto stop = std::chrono::steady_clock::now();
			std::cout << std::format(
				"Loaded in: {:.3f}s\n\n",
				std::chrono::duration<float, std::milli>(stop - start).count() / 1000.0f);
		}

		uint32_t rpp = 1000;
		float timeout = 60.0f;

		// Read config file
		{
			if (config_path.empty())
			{
				std::cout << "No config file given, fallback to defaults.\n\n";
			}
			else
			{
				std::cout << "Reading config file: " << config_path << std::endl;
				const auto start = std::chrono::steady_clock::now();
				std::ifstream file(config_path, std::ios_base::in);
				RZAssert(file.is_open(), "Failed to open file " + config_path.string());
				auto json{nlohmann::json::parse(file, nullptr, true, true)};

				if (json.contains("rpp") && json["rpp"].is_number_unsigned())
					rpp = json["rpp"];
				if (json.contains("timeout") && json["timeout"].is_number_float())
					if (float value{json["timeout"]}; value > 0.0f)
						timeout = value;

				const auto stop = std::chrono::steady_clock::now();
				const auto duration = std::chrono::duration<float, std::milli>(stop - start);
				std::cout << std::format("Configured in: {:.3f}s\n\n", duration.count() / 1000.0f);
			}
		}

		// Render
		{
			engine.renderConfig().tracing().rpp(12);
			const auto start = std::chrono::steady_clock::now();
			std::cout << "Rendering... 0%";
			size_t last_message_length = 0;

			static constexpr std::array stick_array{'|', '/', '-', '\\'};
			int stick_id = 0;
			for (uint32_t traced = 0; traced < rpp;)
			{
				if (rpp - traced < engine.renderConfig().tracing().rpp())
					engine.renderConfig().tracing().rpp(rpp - traced);

				render();
				traced += engine.renderConfig().tracing().rpp();

				const char stick = stick_array[stick_id];
				stick_id = (stick_id + 1) % stick_array.size();

				const auto stop = std::chrono::steady_clock::now();
				const auto duration = std::chrono::duration<float>(stop - start);

				auto message = std::format(
					"\r{} Rendering... {}/{} +{} [rpp] ({:.2f}%) | {:.3f}s (timeout: {:.3f}s)",
					stick,
					traced, rpp,
					engine.renderConfig().tracing().rpp(),
					(traced / float(rpp) * 100.0f),
					duration.count(), timeout);
				std::cout << "\r" << std::string(last_message_length, ' ');
				last_message_length = message.length();
				std::cout << "\r" << message;

				if (duration.count() >= timeout)
					break;
			}
			const auto stop = std::chrono::steady_clock::now();
			std::cout << std::format(
				"\nRendered in: {:.3f}s\n\n",
				std::chrono::duration<float>(stop - start).count());
		}

		// Generate report
		{
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
		}

		return 0;
	}

	void Headless::render()
	{
		auto& engine = RayZath::Engine::Engine::instance();
		auto& world = engine.world();

		// call rendering engine
		const auto start = std::chrono::steady_clock::now();
		engine.renderWorld(true, false);
		const auto stop = std::chrono::steady_clock::now();
		const auto duration = std::chrono::duration<float>(stop - start).count();

		// balance the load
		const float relative_error = (duration - m_load_time) / m_load_time;
		if (std::abs(relative_error) > 0.05f)
		{
			const float duration_ratio = duration / m_load_time;
			const float factor = 1.0f + (1.0f - duration_ratio) * 0.5f;
			m_floaty_rpp *= factor;
			engine.renderConfig().tracing().rpp(std::clamp<uint32_t>(uint32_t(m_floaty_rpp), 1, 1024));
		}
	}
}
