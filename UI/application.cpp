#include "application.hpp"

#include <iostream>

namespace RZ = RayZath::Engine;

namespace RayZath::UI
{
	Application::Application()
		: m_explorer(m_scene, m_viewports)
		, m_main(m_scene)
	{}

	Application& Application::instance()
	{
		static Application application{};
		return application;
	}

	int Application::run()
	{
		m_scene.init();

		return m_rendering.run(
			[this]() { this->update(); },
			[this]() { this->render(); });
	}

	void Application::update()
	{
		static RZ::Timer timer;
		static float ft = 16.66f;

		float elapsed_time = timer.GetTime();
		ft = ft + (elapsed_time - ft) * 0.1f;

		std::stringstream ss;
		ss.precision(2);
		ss << std::fixed << 1000.0f / ft << " fps";
		ss << " (" << std::fixed << ft << "ms)";

		static uint64_t prevRayCount = 0u;
		auto RaysPerSecond = [](uint64_t prevRayCount, uint64_t currRayCount, float ft)
		{
			std::stringstream ss;
			ss << " | ";

			const float rps = ((prevRayCount >= currRayCount) ? currRayCount : (currRayCount - prevRayCount)) * (1000.0f / ft);

			ss.precision(3);
			if (rps > 1.0e9f) ss << rps / 1.0e9f << "G";
			else if (rps > 1.0e6f) ss << rps / 1.0e6f << "M";
			else if (rps > 1.0e3f) ss << rps / 1.0e3f << "K";
			ss << "r/s";

			return ss.str();
		};

		//ss << RaysPerSecond(prevRayCount, m_scene.m_camera->GetRayCount(), ft);
		//prevRayCount = m_scene.m_camera->GetRayCount();
		//m_rendering.setWindowTitle(ss.str());

		m_scene.update(elapsed_time);

		try
		{
			m_scene.render();
		}
		catch (const RayZath::CudaException& ce)
		{
			std::string ce_string = ce.ToString();
			std::cerr << ce_string << std::endl;
		}
		catch (const RayZath::Exception& e)
		{
			std::string e_string = e.ToString();
			std::cerr << e_string << std::endl;
		}

		auto scalePrefix = [](const uint64_t value)
		{
			std::stringstream ss;
			ss.precision(2);
			if (value > 1e12)
				ss << std::fixed << value / 1.0e12 << "T";
			else if (value > 1e9)
				ss << std::fixed << value / 1.0e9 << "G";
			else if (value > 1e6)
				ss << std::fixed << value / 1.0e6 << "M";
			else if (value > 1e3)
				ss << std::fixed << value / 1.0e3 << "K";
			else ss << std::fixed << value;

			return ss.str();
		};
	}
	void Application::render()
	{
		m_viewports.destroyInvalidViewports();

		m_viewports.update(m_rendering.m_vulkan.m_window.currentFrame().commandBuffer());
		m_viewports.draw();

		if (auto selected_camera = m_viewports.getSelected(); selected_camera)
			m_explorer.selectObject<Engine::World::ObjectType::Camera>(selected_camera);

		m_main.update();
		m_explorer.update();
	}
}
