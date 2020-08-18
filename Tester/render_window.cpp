#include "render_window.h"
#include "application.h"

namespace Tester
{
	namespace UI
	{
		RenderWindow::RenderWindow(Interface& interf)
			: mr_iface(interf)
			, mp_camera(mr_iface.mr_app.m_scene.mp_camera)
		{
			// window
			mp_window = WAF::Framework::GetInstance().CreateNewWindow(WAF::ConStruct<WAF::Window>(
				L"Render",
				WAF::Rect(WAF::Point(100, 100), WAF::Size(1280, 720)),
				WAF::Size(300, 200)));
			mp_window->MoveToCenter();

			mp_window->BindEventFunc(&Interface::RenderWindow_OnClose, &mr_iface);
			mp_window->BindEventFunc(&RenderWindow::Window_OnResize, this);

			// graphics box
			mp_gfx_box = mp_window->CreateChild(WAF::ConStruct<WAF::GraphicsBox>(
				WAF::Rect(WAF::Point(0, 0),
					WAF::Size(mp_window->GetClientRect().size.width,
						mp_window->GetClientRect().size.height)),
				WAF::GraphicsBox::GBGraphics::ConStruct(
					WAF::GraphicsBox::RenderType::Hardware,
					WAF::GraphicsBox::PresentOption::RenderImmediately,
					WAF::GraphicsBox::InterpolationMode::Linear,
					WAF::GraphicsBox::TextFormatDescription(L"consolas", 12.0f))));

			mp_gfx_box->BindEventFunc(&RenderWindow::GraphicsBox_OnMouseLPress, this);
			mp_gfx_box->BindEventFunc(&RenderWindow::GraphicsBox_OnMouseMove, this);

			mp_camera->Resize(mp_gfx_box->Gfx.Width, mp_gfx_box->Gfx.Height);
		}
		RenderWindow::~RenderWindow()
		{
			mp_window->Destroy();
		}


		void RenderWindow::BeginDraw()
		{
			mp_gfx_box->Gfx.BeginDraw();
			mp_gfx_box->Gfx.Clear(Graphics::Color(0xFF, 0xFF, 0xFF));
		}
		void RenderWindow::DrawRender(const Graphics::Bitmap& bitmap)
		{
			mp_gfx_box->Gfx.DrawBitmap(
				bitmap,
				Graphics::Rect<float>(0.0f, 0.0f, 
					static_cast<float>(mp_gfx_box->Gfx.Width), static_cast<float>(mp_gfx_box->Gfx.Height)),
				Graphics::Rect<float>(0.0f, 0.0f, static_cast<float>(bitmap.GetWidth()),
					static_cast<float>(bitmap.GetHeight())),
				1.0f,
				WAF::GraphicsBox::InterpolationMode::NearestNeighbor);
		}
		void RenderWindow::DrawDebugInfo(const std::wstring& info)
		{
			mp_gfx_box->Gfx.SetSolidBrush(Graphics::Color(0x00, 0x00, 0x00, 0x80), 0.3f);
			mp_gfx_box->Gfx.FillRectangle(Graphics::Point<float>(0.0f, 0.0f), Graphics::Point<float>(280.0f, 300.0f));

			mp_gfx_box->Gfx.SetSolidBrush(Graphics::Color(255, 255, 255), 1.0f);
			mp_gfx_box->Gfx.DrawString(info, Graphics::Rect<float>(5.0f, 5.0f, 270.0f, 290.0f));
		}
		void RenderWindow::EndDraw()
		{
			mp_gfx_box->Gfx.EndDraw();
		}

		void RenderWindow::UpdateControlKeys(const float elapsed_time)
		{
			RZ::World* MainWorld = &mr_iface.mr_app.m_scene.mr_world;
			Math::vec3<float> curr_pos = mp_camera->GetPosition();
			// x axis
			if (WAF::Framework::GetInstance().Keyboard.KeyPressed(WAF::Keyboard::Key::A))
			{
				mp_camera->SetPosition(Math::vec3<float>(
					curr_pos.x -= 5.0f * cos(mp_camera->GetRotation().y) * elapsed_time,
					curr_pos.y,
					curr_pos.z -= 5.0f * sin(mp_camera->GetRotation().y) * elapsed_time));
			}
			if (WAF::Framework::GetInstance().Keyboard.KeyPressed(WAF::Keyboard::Key::D))
			{
				mp_camera->SetPosition(Math::vec3<float>(
					curr_pos.x += 5.0f * cos(mp_camera->GetRotation().y) * elapsed_time,
					curr_pos.y,
					curr_pos.z += 5.0f * sin(mp_camera->GetRotation().y) * elapsed_time));
			}

			// y axis
			if (WAF::Framework::GetInstance().Keyboard.KeyPressed(WAF::Keyboard::Key::Z))
			{
				mp_camera->SetPosition(Math::vec3<float>(
					curr_pos.x,
					curr_pos.y += 5.0f * elapsed_time,
					curr_pos.z));
			}
			if (WAF::Framework::GetInstance().Keyboard.KeyPressed(WAF::Keyboard::Key::X))
			{
				mp_camera->SetPosition(Math::vec3<float>(
					curr_pos.x,
					curr_pos.y -= 5.0f * elapsed_time,
					curr_pos.z));
			}
			// z axis
			if (WAF::Framework::GetInstance().Keyboard.KeyPressed(WAF::Keyboard::Key::W))
			{
				mp_camera->SetPosition(Math::vec3<float>(
					curr_pos.x += 5.0f * -sin(mp_camera->GetRotation().y) * elapsed_time,
					curr_pos.y,
					curr_pos.z += 5.0f * cos(mp_camera->GetRotation().y) * elapsed_time));
			}
			if (WAF::Framework::GetInstance().Keyboard.KeyPressed(WAF::Keyboard::Key::S))
			{
				mp_camera->SetPosition(Math::vec3<float>(
					curr_pos.x -= 5.0f * -sin(mp_camera->GetRotation().y) * elapsed_time,
					curr_pos.y,
					curr_pos.z -= 5.0f * cos(mp_camera->GetRotation().y) * elapsed_time));
			}
			// rotation:
			// z axis
			if (WAF::Framework::GetInstance().Keyboard.KeyPressed(WAF::Keyboard::Key::Q))
			{
				mp_camera->SetRotation(Math::vec3<float>(
					mp_camera->GetRotation().x,
					mp_camera->GetRotation().y,
					mp_camera->GetRotation().z - 1.0f * elapsed_time));
			}
			if (WAF::Framework::GetInstance().Keyboard.KeyPressed(WAF::Keyboard::Key::E))
			{
				mp_camera->SetRotation(Math::vec3<float>(
					mp_camera->GetRotation().x,
					mp_camera->GetRotation().y,
					mp_camera->GetRotation().z + 1.0f * elapsed_time));
			}
		}

		void RenderWindow::Window_OnResize(WAF::Window::Events::EventResize& event)
		{
			mp_gfx_box->Resize(
				mp_window->GetClientRect().size.width, 
				mp_window->GetClientRect().size.height);

			mr_iface.mr_app.m_scene.ResizeRender(
				mp_gfx_box->Gfx.Width, 
				mp_gfx_box->Gfx.Height);
		}

		void RenderWindow::GraphicsBox_OnMouseLPress(WAF::GraphicsBox::Events::EventMouseLButtonPress& event)
		{
			pressMouseX = mp_gfx_box->GetMousePosition().x;
			pressMouseY = mp_gfx_box->GetMousePosition().y;
			pressCameraRotX = mp_camera->GetRotation().x;
			pressCameraRotY = mp_camera->GetRotation().y;
		}
		void RenderWindow::GraphicsBox_OnMouseMove(WAF::GraphicsBox::Events::EventMouseMove& event)
		{
			if (WAF::Framework::GetInstance().Mouse.LeftPressed)
			{
				mp_camera->SetRotation(
					Math::vec3<float>(
						pressCameraRotX + (pressMouseY - mp_gfx_box->GetMousePosition().y) / 
						300.0f,
						pressCameraRotY + (pressMouseX - mp_gfx_box->GetMousePosition().x) / 
						300.0f,
						mp_camera->GetRotation().z));
			}
		}
	}
}