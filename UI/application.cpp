#include "application.hpp"

namespace RZ = RayZath::Engine;

namespace RayZath::UI
{
	Application::Application() 
		: m_explorer(m_scene)
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
		m_viewport.setCamera(m_scene.m_camera);

		return m_rendering.run([this]() {
			this->update();
			});
	}

	void Application::update()
	{
		static RZ::Timer timer;
		static float ft = 16.66f;

		float elapsed_time = timer.GetTime();
		ft = ft + (elapsed_time - ft) * 0.1f;

		std::wstringstream ss;
		ss.precision(2);
		ss << std::fixed << 1000.0f / ft << " fps";
		ss << " (" << std::fixed << ft << "ms)";

		static uint64_t prevRayCount = 0u;
		auto RaysPerSecond = [](uint64_t prevRayCount, uint64_t currRayCount, float ft)
		{
			std::wstringstream ss;
			ss << " | ";

			const float rps = ((prevRayCount >= currRayCount) ? currRayCount : (currRayCount - prevRayCount)) * (1000.0f / ft);

			ss.precision(3);
			if (rps > 1.0e9f) ss << rps / 1.0e9f << "G";
			else if (rps > 1.0e6f) ss << rps / 1.0e6f << "M";
			else if (rps > 1.0e3f) ss << rps / 1.0e3f << "K";
			ss << "r/s";

			return ss.str();
		};

		ss << RaysPerSecond(prevRayCount, m_scene.m_camera->GetRayCount(), ft);
		prevRayCount = m_scene.m_camera->GetRayCount();

		//m_ui.GetRenderWindow()->mp_window->SetCaption(ss.str());
		//m_ui.GetRenderWindow()->UpdateControlKeys(elapsed_time * 0.001f);

		m_scene.update(elapsed_time);

		//if (m_ui.GetControlPanel()->mp_props_editor)
		//	m_ui.GetControlPanel()->mp_props_editor->UpdateState();

		try
		{
			m_scene.render();
		}
		catch (const RayZath::CudaException& ce)
		{
			std::string ce_string = ce.ToString();
			/*WAF::MessBoxButtonPressed bp = m_ui.GetRenderWindow()->mp_window->ShowMessageBox(
				L"CUDA error",
				std::wstring(ce_string.begin(), ce_string.end()),
				WAF::MessBoxButtonLayout::RetryCancel,
				WAF::MessBoxIcon::Error);

			if (bp == WAF::MessBoxButtonPressed::Cancel)
			{
				m_ui.GetRenderWindow()->mp_window->Close();
				return;
			}*/
		}
		catch (const RayZath::Exception& e)
		{
			std::string e_string = e.ToString();
			/*WAF::MessBoxButtonPressed bp = m_ui.GetRenderWindow()->mp_window->ShowMessageBox(
				L"RZ exception",
				std::wstring(e_string.begin(), e_string.end()),
				WAF::MessBoxButtonLayout::RetryCancel,
				WAF::MessBoxIcon::Error);

			if (bp == WAF::MessBoxButtonPressed::Cancel)
			{
				m_ui.GetRenderWindow()->mp_window->Close();
				return;
			}*/
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

		m_rendering.m_vulkan.m_render_image.updateImage(
			m_scene.getRender(),
			m_rendering.m_vulkan.m_window.currentFrame().commandBuffer());

		m_main.update();
		m_explorer.update();
		m_viewport.update(ft, m_rendering.m_vulkan.m_render_image);
	}
}

/*
#include "application.h"
#include "rayzath.h"

#include <sstream>

namespace Tester
{
	Application::Application()
		: m_scene(*this)
		, m_ui(*this)
		, m_display_info(true)
	{
		WAF::Framework::GetInstance().Keyboard.BindEventFunc(&Application::Keyboard_OnKeyPress, this);
	}
	Application::~Application()
	{

	}

	void Application::Init()
	{
		try
		{
			m_scene.Init();
			m_ui.GetRenderWindow()->SetCamera(m_scene.m_camera);
		}
		catch (const RayZath::Exception& e)
		{
			std::string e_string = e.ToString();
			WAF::MessBoxButtonPressed bp = m_ui.GetRenderWindow()->mp_window->ShowMessageBox(
				L"RZ exception",
				std::wstring(e_string.begin(), e_string.end()),
				WAF::MessBoxButtonLayout::RetryCancel,
				WAF::MessBoxIcon::Error);

			if (bp == WAF::MessBoxButtonPressed::Cancel)
			{
				m_ui.GetRenderWindow()->mp_window->Close();
				return;
			}
		}
	}
	int Application::Start()
	{
		WAF::Framework::GetInstance().SetCallBackFunction(this, &Application::Update);
		return WAF::Framework::GetInstance().ProcessMessages();
	}

	void Application::Update()
	{
		static RZ::Timer timer;
		static float ft = 16.66f;

		float elapsed_time = timer.GetTime();
		ft = ft + (elapsed_time - ft) * 0.1f;

		std::wstringstream ss;
		ss.precision(2);
		ss << std::fixed << 1000.0f / ft << " fps";
		ss << " (" << std::fixed << ft << "ms)";

		static uint64_t prevRayCount = 0u;
		auto RaysPerSecond = [](uint64_t prevRayCount, uint64_t currRayCount, float ft)
		{
			std::wstringstream ss;
			ss << " | ";

			const float rps = ((prevRayCount >= currRayCount) ? currRayCount : (currRayCount - prevRayCount)) * (1000.0f / ft);

			ss.precision(3);
			if (rps > 1.0e9f) ss << rps / 1.0e9f << "G";
			else if (rps > 1.0e6f) ss << rps / 1.0e6f << "M";
			else if (rps > 1.0e3f) ss << rps / 1.0e3f << "K";
			ss << "r/s";

			return ss.str();
		};

		ss << RaysPerSecond(prevRayCount, m_scene.m_camera->GetRayCount(), ft);
		prevRayCount = m_scene.m_camera->GetRayCount();

		m_ui.GetRenderWindow()->mp_window->SetCaption(ss.str());
		m_ui.GetRenderWindow()->UpdateControlKeys(elapsed_time * 0.001f);

		m_scene.Update(elapsed_time);

		if (m_ui.GetControlPanel()->mp_props_editor)
			m_ui.GetControlPanel()->mp_props_editor->UpdateState();

		try
		{
			m_scene.Render();
		}
		catch (const RayZath::CudaException& ce)
		{
			std::string ce_string = ce.ToString();
			WAF::MessBoxButtonPressed bp = m_ui.GetRenderWindow()->mp_window->ShowMessageBox(
				L"CUDA error",
				std::wstring(ce_string.begin(), ce_string.end()),
				WAF::MessBoxButtonLayout::RetryCancel,
				WAF::MessBoxIcon::Error);

			if (bp == WAF::MessBoxButtonPressed::Cancel)
			{
				m_ui.GetRenderWindow()->mp_window->Close();
				return;
			}
		}
		catch (const RayZath::Exception& e)
		{
			std::string e_string = e.ToString();
			WAF::MessBoxButtonPressed bp = m_ui.GetRenderWindow()->mp_window->ShowMessageBox(
				L"RZ exception",
				std::wstring(e_string.begin(), e_string.end()),
				WAF::MessBoxButtonLayout::RetryCancel,
				WAF::MessBoxIcon::Error);

			if (bp == WAF::MessBoxButtonPressed::Cancel)
			{
				m_ui.GetRenderWindow()->mp_window->Close();
				return;
			}
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

		m_ui.GetRenderWindow()->BeginDraw();
		m_ui.GetRenderWindow()->DrawRender(m_scene.GetRender());
		if (m_display_info)
		{
			std::string info = m_scene.mr_engine.GetDebugInfo() +
				"camera:\n " +
				scalePrefix(m_scene.m_camera->GetRayCount()) +
				" rays\n";
			m_ui.GetRenderWindow()->DrawDebugInfo({ info.begin(), info.end() });
		}
		m_ui.GetRenderWindow()->EndDraw();
	}
	void Application::Keyboard_OnKeyPress(WAF::Keyboard::Events::EventKeyPress& event)
	{
		if (event.key == WAF::Keyboard::Key::P)
			m_display_info = !m_display_info;

		if (WAF::Framework::GetInstance().Keyboard.KeyPressed(WAF::Keyboard::Key::Control))
		{
			switch (event.key)
			{
			case WAF::Keyboard::Key::Digit1:
				m_scene.LoadScene(0);
				m_ui.GetRenderWindow()->SetCamera(m_scene.m_camera);
				break;
			case WAF::Keyboard::Key::Digit2:
				m_scene.LoadScene(1);
				m_ui.GetRenderWindow()->SetCamera(m_scene.m_camera);
				break;
			case WAF::Keyboard::Key::Digit3:
				m_scene.LoadScene(2);
				m_ui.GetRenderWindow()->SetCamera(m_scene.m_camera);
				break;
			case WAF::Keyboard::Key::Digit4:
				m_scene.LoadScene(3);
				m_ui.GetRenderWindow()->SetCamera(m_scene.m_camera);
				break;
			case WAF::Keyboard::Key::Digit5:
				m_scene.LoadScene(4);
				m_ui.GetRenderWindow()->SetCamera(m_scene.m_camera);
				break;
			case WAF::Keyboard::Key::Digit6:
				m_scene.LoadScene(5);
				m_ui.GetRenderWindow()->SetCamera(m_scene.m_camera);
				break;
			case WAF::Keyboard::Key::Digit7:
				m_scene.LoadScene(6);
				m_ui.GetRenderWindow()->SetCamera(m_scene.m_camera);
				break;
			}
		}

	}
}
*/
