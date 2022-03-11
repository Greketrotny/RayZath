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

			const float rps = (prevRayCount > currRayCount) ? 0.0f : (currRayCount - prevRayCount) * (1000.0f / ft);

			ss.precision(3);
			if (rps > 1.0e9f) ss << rps / 1.0e9f << "G";
			else if (rps > 1.0e6f) ss << rps / 1.0e6f << "M";
			else if (rps > 1.0e3f) ss << rps / 1.0e3f << "K";
			ss << "rpp";

			return ss.str();
		};

		if (m_scene.m_camera->GetSamplesCount() == 1u) prevRayCount = 0u;
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

		m_ui.GetRenderWindow()->BeginDraw();
		m_ui.GetRenderWindow()->DrawRender(m_scene.GetRender());
		if (m_display_info)
		{
			std::string info = m_scene.mr_engine.GetDebugInfo() +
				"camera:\n " +
				std::to_string(m_scene.m_camera->GetSamplesCount()) +
				"spp\n";
			m_ui.GetRenderWindow()->DrawDebugInfo({ info.begin(), info.end() });
		}
		m_ui.GetRenderWindow()->EndDraw();
	}
	void Application::Keyboard_OnKeyPress(WAF::Keyboard::Events::EventKeyPress& event)
	{
		if (event.key == WAF::Keyboard::Key::P)
			m_display_info = !m_display_info;
		if (event.key == WAF::Keyboard::Key::O)
			m_scene.mr_world.Container<RZ::World::ContainerType::Sphere>().Destroy(0u);

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