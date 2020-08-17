#ifndef RENDER_WINDOW_H
#define RENDER_WINDOW_H

#include "winapi_framework.h"
namespace WAF = WinapiFramework;

namespace Tester
{
	namespace UI
	{
		class Interface;

		class RenderWindow
		{
		private:
			Interface& mr_iface;
		public:
			WAF::Window* mp_window;
			WAF::GraphicsBox* mp_gfx_box;


		public:
			RenderWindow(Interface& interf);
			~RenderWindow();


		public:
			void BeginDraw();
			void DrawRender(const Graphics::Bitmap& bitmap);
			void DrawDebugInfo(const std::wstring& info);
			void EndDraw();


			// ~~~~ event handleers ~~~~
			void Window_OnResize(WAF::Window::Events::EventResize& event);
		};
	}
}

#endif