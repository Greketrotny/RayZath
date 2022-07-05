#include "viewport.hpp"

#include "imgui.h"

#include <iostream>

namespace RZ = RayZath::Engine;

namespace RayZath::UI::Windows
{
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

	Math::vec2f toVec2(const ImVec2 vec)
	{
		return Math::vec2f(vec.x, vec.y);
	}
	ImVec2 toImVec2(const Math::vec2f vec)
	{
		return ImVec2(vec.x, vec.y);
	}

	Viewport::Viewport(RZ::Handle<RZ::Camera> camera, const uint32_t id)
		: m_camera(std::move(camera))
		, m_id(id)
	{}

	void Viewport::update(const Rendering::Vulkan::Handle<VkCommandBuffer>& command_buffer)
	{
		if (m_camera)
		{
			m_image.updateImage(m_camera->GetImageBuffer(), command_buffer);
		}
	}
	void Viewport::draw()
	{
		ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
		ImGui::PushStyleVar(ImGuiStyleVar_WindowMinSize, ImVec2(250, 250));
		if (!ImGui::Begin(("viewport #" + std::to_string(m_id) + "##viewport_window").c_str(), &is_opened,
			ImGuiWindowFlags_NoScrollbar |
			ImGuiWindowFlags_NoCollapse |
			ImGuiWindowFlags_MenuBar |
			ImGuiWindowFlags_NoScrollWithMouse))
		{
			ImGui::End();
			ImGui::PopStyleVar(2);
			return;
		}

		if (ImGui::IsItemClicked(ImGuiMouseButton_Left))
			m_clicked = true;

		drawMenu();
		controlCamera();
		drawRender();

		ImGui::End();
		ImGui::PopStyleVar(2);
	}
	bool Viewport::valid() const
	{
		return is_opened && m_camera;
	}
	bool Viewport::clicked()
	{
		auto c = m_clicked;
		m_clicked = false;
		return c;
	}

	void Viewport::drawMenu()
	{
		ImGui::BeginMenuBar();
		if (ImGui::BeginMenu("canvas"))
		{
			if (ImGui::Button(
				"fit to viewport",
				toImVec2(Math::vec2f(ImGui::GetFontSize()) * Math::vec2f(10.0f, 2.0f))))
				m_fit_to_viewport = true;

			ImGui::SameLine();
			if (ImGui::Button(
				"reset canvas",
				toImVec2(Math::vec2f(ImGui::GetFontSize()) * Math::vec2f(10.0f, 2.0f))))
			{
				m_zoom = 1.0f;
				m_old_image_pos = m_image_pos = Math::vec2f32{};
			}

			ImGui::SetNextItemWidth(200);
			ImGui::SliderFloat("zoom", &m_zoom, 0.5f, 32.0f, nullptr,
				ImGuiSliderFlags_ClampOnInput | ImGuiSliderFlags_Logarithmic);

			ImGui::Checkbox("auto fit", &m_auto_fit);

			ImGui::EndMenu();
		}

		ImGui::EndMenuBar();
	}
	void Viewport::controlCamera()
	{
		if (!m_camera) return;
		const float dt = ImGui::GetIO().DeltaTime;

		// camera resolution
		const auto min = ImGui::GetWindowContentRegionMin();
		const auto max = ImGui::GetWindowContentRegionMax();
		Math::vec2ui32 render_resolution(uint32_t(max.x - min.x), uint32_t(max.y - min.y));
		if (m_camera && m_camera->GetResolution() != render_resolution &&
			(m_auto_fit || m_fit_to_viewport))
		{
			m_camera->Resize(render_resolution);
			m_camera->Focus(m_camera->GetResolution() / 2);

			m_zoom = 1.0f;
			m_old_image_pos = m_image_pos = Math::vec2f{};
			m_fit_to_viewport = false;
		}
		const bool resized = m_previous_resolution != render_resolution;
		m_previous_resolution = render_resolution;

		// camera control
		if (m_camera && ImGui::IsWindowFocused() && !resized && !ImGui::IsKeyDown(ImGuiKey_ModCtrl))
		{
			const float speed = 5.0f;
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

			// roll rotation
			if (ImGui::IsKeyDown(ImGuiKey_E) || ImGui::IsKeyDown(ImGuiKey_Q))
			{
				auto rot = m_camera->GetRotation();
				rot.z += (float(ImGui::IsKeyDown(ImGuiKey_E)) - float(ImGui::IsKeyDown(ImGuiKey_Q))) * dt;
				m_camera->SetRotation(rot);
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
					m_camera->LookAtPoint(m_polar_rotation_origin, m_camera->GetRotation().z);
				}
			}
			// focal point
			else if (ImGui::IsMouseClicked(ImGuiMouseButton_Right || !was_focused))
			{
				m_camera->Focus(Math::vec2ui32(
					std::max(mouse_pos.x, 0),
					std::max(mouse_pos.y, 0)));
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
	}
	void Viewport::drawRender()
	{
		if (!m_image.textureHandle()) return;

		const auto min = ImGui::GetWindowContentRegionMin();
		const auto max = ImGui::GetWindowContentRegionMax();
		const Math::vec2f window_res(max.x - min.x, max.y - min.y);
		const Math::vec2f window_pos(ImGui::GetWindowPos().x + min.x, ImGui::GetWindowPos().y + min.y);
		const auto mouse_pos = toVec2(ImGui::GetMousePos()) - window_pos - window_res / 2;
		Math::vec2f image_res(float(m_image.width()), float(m_image.height()));

		if (ImGui::IsKeyDown(ImGuiKey_ModCtrl))
		{
			if (ImGui::IsMouseClicked(ImGuiMouseButton_Left))
			{
				m_old_image_pos = m_image_pos;
				m_click_pos = mouse_pos;
			}
			else if (ImGui::IsMouseDown(ImGuiMouseButton_Left))
			{
				m_image_pos = m_old_image_pos + (mouse_pos - m_click_pos);
			}

			if (const auto wheel_delta = ImGui::GetIO().MouseWheel; wheel_delta != 0.0f)
			{
				const float new_zoom = std::clamp(m_zoom + wheel_delta * m_zoom * 0.1f, 0.5f, 32.0f);
				const float zoom_factor = new_zoom / m_zoom;
				m_zoom = new_zoom;

				const auto mouse_to_pos = (m_image_pos - mouse_pos) * zoom_factor;
				m_image_pos = mouse_pos + mouse_to_pos;
			}
		}

		image_res *=
			m_zoom *
			std::min(window_res.x / image_res.x, window_res.y / image_res.y);

		m_image_pos.x = std::clamp(m_image_pos.x, -image_res.x / 2 - window_res.x / 2, image_res.x / 2 + window_res.x / 2);
		m_image_pos.y = std::clamp(m_image_pos.y, -image_res.y / 2 - window_res.y / 2, image_res.y / 2 + window_res.y / 2);

		ImGui::SetCursorScreenPos(toImVec2(
			window_pos + window_res / 2 // viewport center
			- image_res / 2	// image half resolution
			+ m_image_pos
		));

		ImGui::Image(
			m_image.textureHandle(),
			toImVec2(image_res));
	}


	Viewport& Viewports::addViewport(RZ::Handle<RZ::Camera> camera)
	{
		// check if viewport for given camera already exists
		for (auto& [viewport_id, viewport] : m_viewports)
			if (viewport.camera() == camera) return viewport;

		// find next free viewport id
		for (uint32_t id = 0; id < m_viewports.size() + 1; id++)
		{
			if (m_viewports.contains(id)) continue;
			auto [element, inserted] = m_viewports.insert(std::make_pair(id, Viewport(std::move(camera), id + 1)));
			RZAssertDebug(inserted, "failed to insert new Viewport");
			return element->second;
		}
		std::terminate();
	}
	void Viewports::destroyInvalidViewports()
	{
		for (auto it = std::begin(m_viewports); it != std::end(m_viewports);)
		{
			if (!it->second.valid()) it = m_viewports.erase(it);
			else it++;
		}
	}

	void Viewports::update(const Rendering::Vulkan::Handle<VkCommandBuffer>& command_buffer)
	{
		for (auto& [id, viewport] : m_viewports)
			viewport.update(command_buffer);
	}
	void Viewports::draw()
	{
		for (auto& [id, viewport] : m_viewports)
			viewport.draw();
	}
}
