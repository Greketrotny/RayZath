#ifndef RENDER_WINDOW_H
#define RENDER_WINDOW_H

#include "winapi_framework.h"
namespace WAF = WinapiFramework;

#include "rayzath.h"
namespace RZ = RayZath::Engine;

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
			Math::vec3f pressCameraPolarRot;
			Math::vec3f polarRotationOrigin = Math::vec3f(0.0f);

			WAF::Point focal_point;


		public:
			RenderWindow(Interface& interf);
			~RenderWindow();


		public:
			void BeginDraw();
			void DrawRender(const Graphics::Bitmap& bitmap);
			void DrawDebugInfo(const std::wstring& info);
			void EndDraw();

			void UpdateControlKeys(const float elapsed_time);
			void SetCamera(RZ::Handle<RZ::Camera> camera);


			// ~~~~ event handleers ~~~~
			// window
			void Window_OnResize(WAF::Window::Events::EventResize& event);

			// graphics box
			void GraphicsBox_OnMouseLPress(WAF::GraphicsBox::Events::EventMouseLButtonPress& event);
			void GraphicsBox_OnMouseRPress(WAF::GraphicsBox::Events::EventMouseRButtonPress& event);
			void GraphicsBox_OnMouseMPress(WAF::GraphicsBox::Events::EventMouseMButtonPress& event);
			void GraphicsBox_OnMouseMove(WAF::GraphicsBox::Events::EventMouseMove& event);
			void GraphicsBox_OnMouseWheel(WAF::GraphicsBox::Events::EventMouseWheel& event);
		};
	}
}

#endif