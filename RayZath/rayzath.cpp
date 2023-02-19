#include "rayzath.hpp"

#include "cuda_engine.cuh"
#include "cpu_engine.hpp"

#include "lib/Json/json.hpp"

#include <chrono>
#include <fstream>
#include <iostream>

namespace RayZath::Engine
{
	Engine::Engine()
		: m_world(std::make_unique<World>())
		, m_cuda_engine(std::make_unique<RayZath::Cuda::Engine>())
		, m_cpu_engine(std::make_unique<RayZath::Engine::CPU::Engine>())
		, m_render_engine(RenderEngine::CUDAGPU)
	{
		srand((unsigned int)(time(NULL)));
	}

	Engine& Engine::instance()
	{
		static Engine engine_instance;
		return engine_instance;
	}
	World& Engine::world()
	{
		return *m_world;
	}
	RenderConfig& Engine::renderConfig()
	{
		return m_render_config;
	}
	Engine::RenderEngine Engine::renderEngine() const
	{
		return m_render_engine;
	}
	void Engine::renderEngine(RenderEngine engine)
	{
		m_render_engine = engine;
	}

	void Engine::renderWorld(
		RenderEngine engine,
		const bool block,
		const bool sync)
	{
		switch (engine)
		{
			case RenderEngine::CUDAGPU:
				m_cuda_engine->renderWorld(*m_world, m_render_config, block, sync);
				break;
			case RenderEngine::CPU:
				m_cpu_engine->renderWorld(*m_world, m_render_config, block, sync);
				break;
			default:
				RZThrowCore("unsupported RenderEngine type");
		}
	}
	void Engine::renderWorld(const bool block, const bool sync)
	{
		renderWorld(m_render_engine, block, sync);
	}

	int Engine::renderWorld(
		std::filesystem::path scene_path, 
		std::filesystem::path report_path, 
		std::filesystem::path config_path)
	{
		using namespace std::chrono_literals;

		// Load scene(s)
		{
			std::cout << "Loading " << scene_path.filename() << std::endl;
			const auto start = std::chrono::steady_clock::now();
			m_world->loader().loadScene(scene_path);
			const auto stop = std::chrono::steady_clock::now();
			std::cout << std::format(
				"Loaded in: {:.3f}s\n\n",
				std::chrono::duration<float, std::milli>(stop - start).count() / 1000.0f);
		}

		size_t rpp = 1000;
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
			renderConfig().tracing().rpp(12);
			const auto start = std::chrono::steady_clock::now();
			std::cout << "Rendering... 0%";
			size_t last_message_length = 0;
			
			const std::array stick_array{'|', '/', '-', '\\'};
			int stick_id = 0;
			for (size_t traced = 0; traced < rpp;)
			{
				if (rpp - traced < renderConfig().tracing().rpp())
					renderConfig().tracing().rpp(uint8_t(rpp - traced));

				renderWorld(true, false);
				traced += renderConfig().tracing().rpp();

				const char stick = stick_array[stick_id];
				stick_id = (stick_id + 1) % stick_array.size();

				const auto stop = std::chrono::steady_clock::now();
				const auto duration = std::chrono::duration<float>(stop - start);

				auto message = std::format(
					"\r{} Rendering... {}/{} [rpp] ({:.2f}%) | {:.3f}s (timeout: {:.3f}s)",
					stick, 
					traced, rpp,
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
				std::chrono::duration<float, std::milli>(stop - start).count() / 1000.0f);
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

			auto& cameras = m_world->container<ObjectType::Camera>();
			for (uint32_t camera_id = 0; camera_id < cameras.count(); camera_id++)
			{
				auto& camera = cameras[camera_id];
				std::cout << "Saving rendered image of \"" << camera->name() << "\"";
				m_world->saver().saveMap<ObjectType::Texture>(
					camera->imageBuffer(), report_path, camera->name());
			}

			const auto stop = std::chrono::steady_clock::now();
			std::cout << std::format(
				"\nGenerated report in: {:.3f}s\n", 
				std::chrono::duration<float, std::milli>(stop - start).count() / 1000.0f);
		}

		return 0;
	}
	

	std::string Engine::debugInfo()
	{
		switch (m_render_engine)
		{
			case RenderEngine::CUDAGPU:
				return m_cuda_engine->timingsString();
			case RenderEngine::CPU:
				return m_cpu_engine->timingsString();
			default:
				RZThrowCore("unsupported RenderEngine type");
		}
	}
}