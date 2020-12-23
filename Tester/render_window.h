#ifndef RENDER_WINDOW_H
#define RENDER_WINDOW_H

#include "winapi_framework.h"
namespace WAF = WinapiFramework;

#include "rayzath.h"
namespace RZ = RayZath;

namespace Tester
{
	namespace UI
	{
		class Interface;

		class RenderWindow
		{
		private:
			Interface& mr_iface;
			RZ::Handle<RZ::Camera> m_camera;
		public:
			WAF::Window* mp_window;
			WAF::GraphicsBox* mp_gfx_box;

			int pressMouseX, pressMouseY;
			float pressCameraRotX, pressCameraRotY;


		public:
			RenderWindow(Interface& interf);
			~RenderWindow();


		public:
			void BeginDraw();
			void DrawRender(const Graphics::Bitmap& bitmap);
			void DrawDebugInfo(const std::wstring& info);
			void EndDraw();

			void UpdateControlKeys(const float elapsed_time);


			// ~~~~ event handleers ~~~~
			// window
			void Window_OnResize(WAF::Window::Events::EventResize& event);

			// graphics box
			void GraphicsBox_OnMouseLPress(WAF::GraphicsBox::Events::EventMouseLButtonPress& event);
			void GraphicsBox_OnMouseMove(WAF::GraphicsBox::Events::EventMouseMove& event);
		};
	}
}

#endif