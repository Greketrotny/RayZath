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
		m_scene.Render();

		Graphics::Bitmap b(50, 50);
		for (int x = 0; x < b.GetWidth(); x++)
		{
			for (int y = 0; y < b.GetHeight(); y++)
			{
				b.SetPixel(x, y, Graphics::Color(
					x / static_cast<float>(b.GetWidth()) * 255.0f,
					y / static_cast<float>(b.GetHeight()) * 255.0f,
					0x00));
			}
		}
		b.SetPixel(10, 10, Graphics::Color(0xFF, 0xFF, 0xFF));

		m_ui.GetRenderWindow()->BeginDraw();
		m_ui.GetRenderWindow()->DrawRender(b);
		m_ui.GetRenderWindow()->EndDraw();

		Sleep(1);
	}
}