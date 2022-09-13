#include "application.hpp"

#include "rzexception.hpp"
#include "cuda_exception.hpp"

#include <iostream>

namespace RZ = RayZath::Engine;

namespace RayZath::UI
{
	Application::Application()
		: m_main_window(m_scene, m_rendering)
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
		float elapsed_time = timer.time();
		m_scene.update(elapsed_time);

		try
		{
			m_scene.render();
		}
		catch (const RayZath::Cuda::Exception& ce)
		{
			std::cerr << ce.what()  << std::endl;
		}
		catch (const RayZath::Exception& e)
		{
			std::cerr << e.what() << std::endl;
		}
	}
	void Application::render()
	{
		m_main_window.update();
	}
}
