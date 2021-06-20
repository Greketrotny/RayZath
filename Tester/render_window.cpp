#include "render_window.h"
#include "application.h"

namespace Tester
{
	namespace UI
	{
		RenderWindow::RenderWindow(Interface& interf)
			: mr_iface(interf)
			, m_camera(mr_iface.mr_app.m_scene.m_camera)
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
					WAF::GraphicsBox::PresentOption::RenderImmediately,
					WAF::GraphicsBox::InterpolationMode::Linear,
					WAF::GraphicsBox::TextFormatDescription(L"consolas", 12.0f))));

			mp_gfx_box->BindEventFunc(&RenderWindow::GraphicsBox_OnMouseLPress, this);
			mp_gfx_box->BindEventFunc(&RenderWindow::GraphicsBox_OnMouseRPress, this);
			mp_gfx_box->BindEventFunc(&RenderWindow::GraphicsBox_OnMouseMPress, this);
			mp_gfx_box->BindEventFunc(&RenderWindow::GraphicsBox_OnMouseMove, this);
			mp_gfx_box->BindEventFunc(&RenderWindow::GraphicsBox_OnMouseWheel, this);

			m_camera->Resize(Math::vec2ui32(mp_gfx_box->Gfx.Width, mp_gfx_box->Gfx.Height));
			focal_point = WAF::Point(m_camera->GetWidth() / 2u, m_camera->GetHeight() / 2u);
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
			mp_gfx_box->Gfx.FillRectangle(Graphics::Point<float>(0.0f, 0.0f), Graphics::Point<float>(260.0f, 320.0f));

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
			Math::vec3f curr_pos = m_camera->GetPosition();
			// x axis
			if (WAF::Framework::GetInstance().Keyboard.KeyPressed(WAF::Keyboard::Key::A))
			{
				m_camera->SetPosition(Math::vec3f(
					curr_pos.x -= 5.0f * cos(m_camera->GetRotation().y) * elapsed_time,
					curr_pos.y,
					curr_pos.z -= 5.0f * sin(m_camera->GetRotation().y) * elapsed_time));
			}
			if (WAF::Framework::GetInstance().Keyboard.KeyPressed(WAF::Keyboard::Key::D))
			{
				m_camera->SetPosition(Math::vec3f(
					curr_pos.x += 5.0f * cos(m_camera->GetRotation().y) * elapsed_time,
					curr_pos.y,
					curr_pos.z += 5.0f * sin(m_camera->GetRotation().y) * elapsed_time));
			}

			// y axis
			if (WAF::Framework::GetInstance().Keyboard.KeyPressed(WAF::Keyboard::Key::Z))
			{
				m_camera->SetPosition(Math::vec3f(
					curr_pos.x,
					curr_pos.y += 5.0f * elapsed_time,
					curr_pos.z));
			}
			if (WAF::Framework::GetInstance().Keyboard.KeyPressed(WAF::Keyboard::Key::X))
			{
				m_camera->SetPosition(Math::vec3f(
					curr_pos.x,
					curr_pos.y -= 5.0f * elapsed_time,
					curr_pos.z));
			}
			// z axis
			if (WAF::Framework::GetInstance().Keyboard.KeyPressed(WAF::Keyboard::Key::W))
			{
				m_camera->SetPosition(Math::vec3f(
					curr_pos.x += 5.0f * -sin(m_camera->GetRotation().y) * elapsed_time,
					curr_pos.y,
					curr_pos.z += 5.0f * cos(m_camera->GetRotation().y) * elapsed_time));
			}
			if (WAF::Framework::GetInstance().Keyboard.KeyPressed(WAF::Keyboard::Key::S))
			{
				m_camera->SetPosition(Math::vec3f(
					curr_pos.x -= 5.0f * -sin(m_camera->GetRotation().y) * elapsed_time,
					curr_pos.y,
					curr_pos.z -= 5.0f * cos(m_camera->GetRotation().y) * elapsed_time));
			}
			// rotation:
			// z axis
			if (WAF::Framework::GetInstance().Keyboard.KeyPressed(WAF::Keyboard::Key::Q))
			{
				m_camera->SetRotation(Math::vec3f(
					m_camera->GetRotation().x,
					m_camera->GetRotation().y,
					m_camera->GetRotation().z - 1.0f * elapsed_time));
			}
			if (WAF::Framework::GetInstance().Keyboard.KeyPressed(WAF::Keyboard::Key::E))
			{
				m_camera->SetRotation(Math::vec3f(
					m_camera->GetRotation().x,
					m_camera->GetRotation().y,
					m_camera->GetRotation().z + 1.0f * elapsed_time));
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
			pressCameraRotX = m_camera->GetRotation().x;
			pressCameraRotY = m_camera->GetRotation().y;
		}
		void RenderWindow::GraphicsBox_OnMouseRPress(WAF::GraphicsBox::Events::EventMouseRButtonPress& event)
		{
			focal_point = mp_gfx_box->GetMousePosition();
			m_camera->Focus(Math::vec2ui32(focal_point.x, focal_point.y));
		}

		Math::vec3f PolarRotation(const Math::vec3f& v)
		{
			const float theta = acosf(v.Normalized().y);
			const float phi = atan2f(v.z, v.x);
			return { theta, phi, v.Magnitude() };
		}
		Math::vec3f CartesianDirection(const Math::vec3f& polar)
		{
			return Math::vec3f(cosf(polar.y) * sinf(polar.x), cosf(polar.x), sinf(polar.y) * sinf(polar.x)) * polar.z;
		}
		void RenderWindow::GraphicsBox_OnMouseMPress(WAF::GraphicsBox::Events::EventMouseMButtonPress& event)
		{
			pressMouseX = mp_gfx_box->GetMousePosition().x;
			pressMouseY = mp_gfx_box->GetMousePosition().y;

			Math::vec3f to_camera = m_camera->GetPosition() - polarRotationOrigin;
			pressCameraPolarRot = PolarRotation(to_camera);
		}
		void RenderWindow::GraphicsBox_OnMouseMove(WAF::GraphicsBox::Events::EventMouseMove& event)
		{
			if (WAF::Framework::GetInstance().Mouse.LeftPressed)
			{
				m_camera->SetRotation(
					Math::vec3f(
						pressCameraRotX + 
						(pressMouseY - mp_gfx_box->GetMousePosition().y) / 300.0f,
						pressCameraRotY + 
						(pressMouseX - mp_gfx_box->GetMousePosition().x) / 300.0f,
						m_camera->GetRotation().z));
			}
			if (WAF::Framework::GetInstance().Mouse.RightPressed)
			{
				focal_point = mp_gfx_box->GetMousePosition();
				m_camera->Focus(Math::vec2ui32(focal_point.x, focal_point.y));
			}
			if (WAF::Framework::GetInstance().Mouse.MiddlePressed)
			{
				m_camera->SetPosition(
					polarRotationOrigin +
					CartesianDirection(Math::vec3f(pressCameraPolarRot.x +
						(pressMouseY - mp_gfx_box->GetMousePosition().y) / 300.0f,
						pressCameraPolarRot.y +
						(pressMouseX - mp_gfx_box->GetMousePosition().x) / 300.0f, pressCameraPolarRot.z)));

				m_camera->LookAtPoint(polarRotationOrigin);
			}
		}
		void RenderWindow::GraphicsBox_OnMouseWheel(WAF::GraphicsBox::Events::EventMouseWheel& event)
		{
			Math::vec3f OC = m_camera->GetPosition() - polarRotationOrigin;
			const float step = 20.0f;
			OC *= (step - event.delta) / step;
			m_camera->SetPosition(polarRotationOrigin + OC);
		}
	}
}