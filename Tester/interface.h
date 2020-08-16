#ifndef INTERFACE_H
#define INTERFACE_H

#include "winapi_framework.h"

#include "control_panel.h"
#include "render_window.h"

namespace Tester
{
	class Application;

	namespace UI
	{
		class Interface
		{
		private:
			Application& mr_app;

			ControlPanel* mp_control_panel;
			RenderWindow* mp_render_window;


		public:
			Interface(Application& app);
			~Interface();


		public:
			ControlPanel* GetControlPanel();
			RenderWindow* GetRenderWindow();


		public:
			// ~~~~ event handlers ~~~~
			void RenderWindow_OnClose(WAF::Window::Events::EventClose& event);
			void ControlPanel_OnClose(WAF::Window::Events::EventClose& event);
		};
	}
}


#endif // !INTERFACE_H