#include "interface.h"
#include "application.h"

namespace Tester
{
	namespace UI
	{
		Interface::Interface(Application& app)
			: mr_app(app)
		{
			mp_control_panel = new ControlPanel(*this);
			mp_render_window = new RenderWindow(*this);
		}
		Interface::~Interface()
		{
			if (mp_control_panel) delete mp_control_panel;
			if (mp_render_window) delete mp_render_window;
		}

		ControlPanel* Interface::GetControlPanel()
		{
			return mp_control_panel;
		}
		RenderWindow* Interface::GetRenderWindow()
		{
			return mp_render_window;
		}


		void Interface::RenderWindow_OnClose(WAF::Window::Events::EventClose& event)
		{
			event.AbortClosing();
			WAF::Framework::GetInstance().Exit(0);

			if (mp_control_panel)
			{
				delete mp_control_panel;
				mp_control_panel = nullptr;
			}

			if (mp_render_window)
			{
				delete mp_render_window;
				mp_render_window = nullptr;
			}
		}
		void Interface::ControlPanel_OnClose(WAF::Window::Events::EventClose& event)
		{
			event.AbortClosing();
			WAF::Framework::GetInstance().Exit(0);

			if (mp_control_panel)
			{
				delete mp_control_panel;
				mp_control_panel = nullptr;
			}

			if (mp_render_window)
			{
				delete mp_render_window;
				mp_render_window = nullptr;
			}
		}
	}
}