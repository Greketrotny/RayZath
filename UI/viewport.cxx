module;

#include "imgui.h"

#include "rayzath.h"

#include <iostream>

module rz.ui.windows.viewport;

namespace RZ = RayZath::Engine;

namespace RayZath::UI
{
	void Viewport::setCamera(RZ::Handle<RZ::Camera> camera)
	{
		m_camera = std::move(camera);
	}

	Math::vec3f polarRotation(const Math::vec3f& v)
	{
		const float theta = acosf(v.Normalized().y);
		const float phi = atan2f(v.z, v.x);
		return { theta, phi, v.Magnitude() };
	}
	Math::vec3f cartesianDirection(const Math::vec3f& polar)
	{
		return Math::vec3f(cosf(polar.y) * sinf(polar.x), cosf(polar.x), sinf(polar.y) * sinf(polar.x)) * polar.z;
	}

	void Viewport::update(const float dt, const Rendering::Vulkan::Image& image)
	{
		ImGui::PushStyleVar(ImGuiStyleVar_::ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
		ImGui::Begin("viewport", nullptr,
			ImGuiWindowFlags_NoScrollbar |
			ImGuiWindowFlags_NoCollapse);

		// camera resolution

		const auto min = ImGui::GetWindowContentRegionMin();
		const auto max = ImGui::GetWindowContentRegionMax();
		Math::vec2ui32 render_resolution(uint32_t(max.x - min.x), uint32_t(max.y - min.y));
		if (m_camera && m_camera->GetResolution() != render_resolution)
		{
			m_camera->Resize(render_resolution);
			m_camera->Focus(m_camera->GetResolution() / 2);
		}
		const bool resized = m_previous_resolution != render_resolution;
		m_previous_resolution = render_resolution;

		// camera control
		if (m_camera && ImGui::IsWindowFocused() && !resized)
		{
			const float speed = 0.005f;
			const float rotation_speed = 0.004f;
			const float zoom_speed = 5.0f;

			// camera velocity
			Math::vec3f velocity = Math::vec3f(
				float(ImGui::IsKeyDown(ImGuiKey_D)) - float(ImGui::IsKeyDown(ImGuiKey_A)),
				float(ImGui::IsKeyDown(ImGuiKey_Z)) - float(ImGui::IsKeyDown(ImGuiKey_X)),
				float(ImGui::IsKeyDown(ImGuiKey_W)) - float(ImGui::IsKeyDown(ImGuiKey_S))) *
				dt * speed;
			if (velocity.x != 0.0f || velocity.y != 0.0f || velocity.z != 0.0f)
			{
				m_camera->SetPosition(
					m_camera->GetPosition() +
					m_camera->GetCoordSystem().GetXAxis() * velocity.x +
					m_camera->GetCoordSystem().GetYAxis() * velocity.y +
					m_camera->GetCoordSystem().GetZAxis() * velocity.z);
			}


			// camera rotation
			const Math::vec2i32 mouse_pos(
				int32_t(ImGui::GetMousePos().x - ImGui::GetWindowPos().x),
				int32_t(ImGui::GetMousePos().y - ImGui::GetWindowPos().y));
			if (ImGui::IsMouseClicked(ImGuiMouseButton_Left) || !was_focused)
			{
				m_mouse_click_position = m_mouse_previous_position = mouse_pos;
				m_mouse_click_rotation.x = m_camera->GetRotation().x;
				m_mouse_click_rotation.y = m_camera->GetRotation().y;
			}
			else if (ImGui::IsMouseDown(ImGuiMouseButton_Left))
			{
				if (mouse_pos != m_mouse_previous_position)
				{
					m_mouse_previous_position = mouse_pos;
					m_camera->SetRotation(
						Math::vec3f(
							m_mouse_click_rotation.x +
							(m_mouse_click_position.y - mouse_pos.y) * rotation_speed,
							m_mouse_click_rotation.y +
							(m_mouse_click_position.x - mouse_pos.x) * rotation_speed,
							m_camera->GetRotation().z));
				}
			}
			// focal point
			else if (ImGui::IsMouseClicked(ImGuiMouseButton_Right || !was_focused))
			{
				m_camera->Focus(Math::vec2ui32(
					std::max(mouse_pos.x, 0),
					std::max(mouse_pos.y, 0)));
			}
			// polar rotation
			else if (ImGui::IsMouseClicked(ImGuiMouseButton_Middle) || !was_focused)
			{
				m_mouse_click_position = m_mouse_previous_position = mouse_pos;
				Math::vec3f to_camera = m_camera->GetPosition() - m_polar_rotation_origin;
				m_mouse_click_polar_rotation = polarRotation(to_camera);
			}
			else if (ImGui::IsMouseDown(ImGuiMouseButton_Middle))
			{
				if (mouse_pos != m_mouse_previous_position)
				{
					m_mouse_previous_position = mouse_pos;
					m_camera->SetPosition(
						m_polar_rotation_origin +
						cartesianDirection(Math::vec3f(
							m_mouse_click_polar_rotation.x +
							(m_mouse_click_position.y - mouse_pos.y) * rotation_speed,
							m_mouse_click_polar_rotation.y +
							(m_mouse_click_position.x - mouse_pos.x) * rotation_speed,
							m_mouse_click_polar_rotation.z)));
					m_camera->LookAtPoint(m_polar_rotation_origin);
				}
			}

			// zoom
			if (const auto wheel = ImGui::GetIO().MouseWheel; wheel != 0.0f)
			{
				auto OC = m_camera->GetPosition() - m_polar_rotation_origin;
				const float step = 100.0f / zoom_speed;
				OC *= (step - std::min(wheel, step * 0.5f)) / step;
				m_camera->SetPosition(m_polar_rotation_origin + OC);
			}



			was_focused = true;
		}
		else
		{
			was_focused = false;
		}



		if (image.textureHandle())
		{
			ImGui::Image(image.textureHandle(), ImVec2(float(image.width()), float(image.height())));
		}


		/*ImGui::GetWindowDrawList()->AddCircle(
			ImVec2(m_mouse_click_position.x, m_mouse_click_position.y),
			10.0f, 0xFFFFFFFF, 10, 1.0f);*/

			/*
			ImTextureID my_tex_id = io.Fonts->TexID;
			float my_tex_w = (float)io.Fonts->TexWidth;
			float my_tex_h = (float)io.Fonts->TexHeight;
			{
				ImGui::Text("%.0fx%.0f", my_tex_w, my_tex_h);
				ImVec2 pos = ImGui::GetCursorScreenPos();
				ImVec2 uv_min = ImVec2(0.0f, 0.0f);                 // Top-left
				ImVec2 uv_max = ImVec2(1.0f, 1.0f);                 // Lower-right
				ImVec4 tint_col = ImVec4(1.0f, 1.0f, 1.0f, 1.0f);   // No tint
				ImVec4 border_col = ImVec4(1.0f, 1.0f, 1.0f, 0.5f); // 50% opaque white
				ImGui::Image(my_tex_id, ImVec2(my_tex_w, my_tex_h), uv_min, uv_max, tint_col, border_col);
				if (ImGui::IsItemHovered())
				{
					ImGui::BeginTooltip();
					float region_sz = 32.0f;
					float region_x = io.MousePos.x - pos.x - region_sz * 0.5f;
					float region_y = io.MousePos.y - pos.y - region_sz * 0.5f;
					float zoom = 4.0f;
					if (region_x < 0.0f) { region_x = 0.0f; }
					else if (region_x > my_tex_w - region_sz) { region_x = my_tex_w - region_sz; }
					if (region_y < 0.0f) { region_y = 0.0f; }
					else if (region_y > my_tex_h - region_sz) { region_y = my_tex_h - region_sz; }
					ImGui::Text("Min: (%.2f, %.2f)", region_x, region_y);
					ImGui::Text("Max: (%.2f, %.2f)", region_x + region_sz, region_y + region_sz);
					ImVec2 uv0 = ImVec2((region_x) / my_tex_w, (region_y) / my_tex_h);
					ImVec2 uv1 = ImVec2((region_x + region_sz) / my_tex_w, (region_y + region_sz) / my_tex_h);
					ImGui::Image(my_tex_id, ImVec2(region_sz * zoom, region_sz * zoom), uv0, uv1, tint_col, border_col);
					ImGui::EndTooltip();
				}
			}
			*/

		ImGui::End();
		ImGui::PopStyleVar();

		ImGui::ShowDemoWindow();
	}
}
