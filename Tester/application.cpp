#include "application.h"

namespace Tester
{
	Application::Application()
		: m_ui(*this)
		, m_scene(*this)
	{
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

		std::wstringstream ss;
		ss.precision(3);
		ss << "Frame time: " << std::fixed << timer.GetTime() << "ms";
		m_ui.GetRenderWindow()->mp_window->SetCaption(ss.str());

		if (WAF::Framework::GetInstance().Keyboard.KeyPressed(WAF::Keyboard::Key::U))
			m_scene.mr_world.RequestUpdate();

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

		m_ui.GetRenderWindow()->BeginDraw();
		m_ui.GetRenderWindow()->DrawRender(m_scene.GetRender());
		m_ui.GetRenderWindow()->DrawDebugInfo(
			m_scene.mr_engine.mp_cuda_engine->mainDebugInfo.InfoToString());
		m_ui.GetRenderWindow()->EndDraw();
	}
}