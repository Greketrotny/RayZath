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
		m_ui.GetRenderWindow()->mp_window->SetCaption(ss.str());
		m_ui.GetRenderWindow()->UpdateControlKeys(elapsed_time * 0.001f);

		m_scene.Update(elapsed_time);

		if (m_ui.GetControlPanel()->mp_props_editor)
			m_ui.GetControlPanel()->mp_props_editor->UpdateState();

		try
		{
			m_scene.Render();
		}
		catch (const RZ::CudaException& ce)
		{
			WAF::MessBoxButtonPressed bp = m_ui.GetRenderWindow()->mp_window->ShowMessageBox(
				L"CUDA error",
				ce.ToString(),
				WAF::MessBoxButtonLayout::RetryCancel,
				WAF::MessBoxIcon::Error);

			if (bp == WAF::MessBoxButtonPressed::Cancel)
			{
				m_ui.GetRenderWindow()->mp_window->Close();
				return;
			}
		}
		catch (const RZ::Exception& e)
		{
			WAF::MessBoxButtonPressed bp = m_ui.GetRenderWindow()->mp_window->ShowMessageBox(
				L"RZ exception",
				e.ToString(),
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
		if (m_display_info) m_ui.GetRenderWindow()->DrawDebugInfo(
			m_scene.mr_engine.GetDebugInfo());
		m_ui.GetRenderWindow()->EndDraw();
	}
	void Application::Keyboard_OnKeyPress(WAF::Keyboard::Events::EventKeyPress& event)
	{
		if (event.key == WAF::Keyboard::Key::P)
			m_display_info = !m_display_info;
		if (event.key == WAF::Keyboard::Key::O)
			m_scene.mr_world.GetSpheres().Destroy(0u);
	}
}