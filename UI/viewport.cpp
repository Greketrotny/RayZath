#include "viewport.hpp"

#include "text_utils.h"
#include "imgui.h"

namespace RZ = RayZath::Engine;

namespace RayZath::UI::Windows
{
	Math::vec3f polarRotation(const Math::vec3f& v)
	{
		const float theta = acosf(v.Normalized().y);
		const float phi = atan2f(v.z, v.x);
		return {theta, phi, v.Magnitude()};
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

	Viewport::Viewport(
		std::reference_wrapper<RZ::World> world,
		RZ::Handle<RZ::Camera> camera,
		const uint32_t id)
		: mr_world(world)
		, m_camera(std::move(camera))
		, m_id(id)
	{}

	bool Viewport::valid() const
	{
		return m_is_opened && m_camera;
	}
	bool Viewport::clicked()
	{
		auto c = m_clicked;
		m_clicked = false;
		return c;
	}
	bool Viewport::selected()
	{
		auto s = m_selected;
		m_selected = false;
		return s;
	}

	void Viewport::update(const Rendering::Vulkan::Handle<VkCommandBuffer>& command_buffer)
	{
		const auto window_screen_pos = toVec2(ImGui::GetWindowPos());
		m_content_min = toVec2(ImGui::GetWindowContentRegionMin()) + window_screen_pos;
		m_content_max = toVec2(ImGui::GetWindowContentRegionMax()) + window_screen_pos;
		Math::vec2i32 screen_mouse_pos(toVec2(ImGui::GetMousePos()));
		m_content_mouse_pos = screen_mouse_pos - m_content_min;
		m_content_res = m_content_max - m_content_min;
		m_resized = m_prev_content_res != m_content_res;
		m_prev_content_res = m_content_res;

		if (m_camera)
		{
			m_image.updateImage(m_camera->imageBuffer(), command_buffer);
		}
	}
	void Viewport::draw(const Rendering::Vulkan::Handle<VkCommandBuffer>& command_buffer)
	{
		ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
		ImGui::PushStyleVar(ImGuiStyleVar_WindowMinSize, ImVec2(250, 250));
		const std::string title =
			(m_camera ? m_camera->name() : "empty") +
			"###viewport_id" + std::to_string(m_id);
		if (!ImGui::Begin(title.c_str(), &m_is_opened,
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

		update(command_buffer);
		drawMenu();
		controlCamera();
		controlCanvas();
		drawRender();

		ImGui::End();
		ImGui::PopStyleVar(2);

		drawStats();
	}

	void Viewport::drawMenu()
	{
		ImGui::BeginMenuBar();
		if (ImGui::BeginMenu("canvas"))
		{
			if (ImGui::Button(
				"fit to viewport",
				toImVec2(Math::vec2f(ImGui::GetFontSize()) * Math::vec2f(10.0f, 0.0f))))
				m_fit_to_viewport = true;

			ImGui::SameLine();
			if (ImGui::Button(
				"reset canvas",
				toImVec2(Math::vec2f(ImGui::GetFontSize()) * Math::vec2f(10.0f, 0.0f))))
				m_reset_canvas = true;

			ImGui::SetNextItemWidth(200);
			ImGui::SliderFloat("zoom", &m_zoom, 0.5f, 32.0f, nullptr,
				ImGuiSliderFlags_ClampOnInput | ImGuiSliderFlags_Logarithmic);

			ImGui::Checkbox("auto fit", &m_auto_fit);

			ImGui::EndMenu();
		}
		if (ImGui::BeginMenu("animate"))
		{
			if (ImGui::Checkbox("animate", &m_animate))
			{
				m_rotation_vector = m_camera->position() - m_rotation_center;
			}
			float origin[3] = {m_rotation_center.x, m_rotation_center.y, m_rotation_center.z};
			if (ImGui::DragFloat3("rotation origin", origin, 0.01f))
				m_rotation_center = Math::vec3f32(origin[0], origin[1], origin[2]);
			ImGui::DragFloat("rotation speed", &m_rotation_speed, 0.01f);

			ImGui::EndMenu();
		}
		if (ImGui::BeginMenu("save"))
		{
			auto save = [&]()
			{
				if (m_camera)
				{
					mr_world.get().saver().saveMap<RZ::World::ObjectType::Texture>(
						m_camera->imageBuffer(),
						std::filesystem::path(std::string_view{m_path_buffer.data()}),
						m_camera->name() + '_' + Utils::scientificWithPrefix(m_camera->rayCount()) + "_rays");
				}
				m_error_message.clear();
			};

			try
			{
				if (ImGui::InputTextWithHint(
					"##path_input", "save path",
					m_path_buffer.data(), m_path_buffer.size(),
					ImGuiInputTextFlags_EnterReturnsTrue))
				{
					save();
				}
				ImGui::SameLine();
				if (ImGui::Button("save"))
				{
					save();
				}
			}
			catch (std::exception& e)
			{
				m_error_message = e.what();
			}

			if (!m_error_message.empty())
			{
				ImGui::SetNextItemWidth(300.0f);
				ImGui::PushStyleColor(ImGuiCol_::ImGuiCol_Text, ImVec4(1.0f, 0.2f, 0.2f, 1.0f));
				ImGui::TextWrapped("%s", m_error_message.c_str());
				ImGui::PopStyleColor();
			}

			ImGui::EndMenu();
		}
		if (ImGui::BeginMenu("stats"))
		{
			if (ImGui::MenuItem("render statistics"))
			{
				if (!m_stats)
					m_stats = std::make_unique<Stats>();
			}
			ImGui::EndMenu();
		}

		ImGui::EndMenuBar();
	}
	void Viewport::controlCamera()
	{
		if (!m_camera) return;
		const float dt = ImGui::GetIO().DeltaTime;

		if (m_camera->enabled())
		{
			// adjust camera resolution
			if (Math::vec2i32(m_camera->resolution()) != m_content_res && (m_auto_fit || m_fit_to_viewport))
			{
				Math::vec2u32 new_res(std::max(m_content_res.x, 1), std::max(m_content_res.y, 1));
				m_camera->resize(new_res);
				m_camera->focus(m_camera->resolution() / 2);
				m_fit_to_viewport = false;
			}

			if ((m_selected_mesh != m_camera->m_raycasted_mesh ||
				m_selected_material != m_camera->m_raycasted_material) &&
				m_requested_select)
			{
				m_selected_mesh = m_camera->m_raycasted_mesh;
				m_selected_material = m_camera->m_raycasted_material;
				m_selected = true;
				m_requested_select = false;
			}

			// camera control
			if (ImGui::IsWindowFocused() && !m_resized && !ImGui::IsKeyDown(ImGuiKey_ModCtrl))
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
					m_camera->position(
						m_camera->position() +
						m_camera->coordSystem().xAxis() * velocity.x +
						m_camera->coordSystem().yAxis() * velocity.y +
						m_camera->coordSystem().zAxis() * velocity.z);

					if (m_animate)
					{
						m_rotation_vector = m_camera->position() - m_rotation_center;
						m_rotation_angle = 0.0f;
					}
				}

				// roll rotation
				if (ImGui::IsKeyDown(ImGuiKey_E) || ImGui::IsKeyDown(ImGuiKey_Q))
				{
					auto rot{m_camera->rotation()};
					rot.z += (float(ImGui::IsKeyDown(ImGuiKey_E)) - float(ImGui::IsKeyDown(ImGuiKey_Q))) * dt;
					m_camera->rotation(rot);
				}

				// camera rotation
				if ((ImGui::IsMouseClicked(ImGuiMouseButton_Left) || !m_was_focused) &&
					m_content_mouse_pos.x >= 0 && m_content_mouse_pos.y >= 0 &&
					m_content_mouse_pos.x < m_content_res.x && m_content_mouse_pos.y < m_content_res.y)
				{
					if (m_was_focused && m_camera->getRayCastPixel() != Math::vec2ui32(m_image_click_pos))
					{
						m_requested_select = true;
						m_camera->rayCastPixel(Math::vec2ui32(m_image_click_pos));
					}					

					m_mouse_dragging = true;
					m_content_mouse_click_pos = m_content_mouse_prev_pos = m_content_mouse_pos;
					m_mouse_click_rotation.x = m_camera->rotation().x;
					m_mouse_click_rotation.y = m_camera->rotation().y;
				}
				else if (ImGui::IsMouseDown(ImGuiMouseButton_Left) && m_mouse_dragging)
				{
					if (m_content_mouse_pos != m_content_mouse_prev_pos)
					{
						m_content_mouse_prev_pos = m_content_mouse_pos;
						m_camera->rotation(
							Math::vec3f(
								m_mouse_click_rotation.x +
								(m_content_mouse_click_pos.y - m_content_mouse_pos.y) * rotation_speed,
								m_mouse_click_rotation.y +
								(m_content_mouse_click_pos.x - m_content_mouse_pos.x) * rotation_speed,
								m_camera->rotation().z));
					}
				}
				// polar rotation
				else if (ImGui::IsMouseClicked(ImGuiMouseButton_Middle) || !m_was_focused)
				{
					m_content_mouse_click_pos = m_content_mouse_prev_pos = m_content_mouse_pos;
					Math::vec3f to_camera = m_camera->position() - m_polar_rotation_origin;
					m_mouse_click_polar_rotation = polarRotation(to_camera);
				}
				else if (ImGui::IsMouseDown(ImGuiMouseButton_Middle))
				{
					if (m_content_mouse_pos != m_content_mouse_prev_pos)
					{
						m_content_mouse_prev_pos = m_content_mouse_pos;
						m_camera->position(
							m_polar_rotation_origin +
							cartesianDirection(Math::vec3f(
								m_mouse_click_polar_rotation.x +
								(m_content_mouse_click_pos.y - m_content_mouse_pos.y) * rotation_speed,
								m_mouse_click_polar_rotation.y +
								(m_content_mouse_click_pos.x - m_content_mouse_pos.x) * rotation_speed,
								m_mouse_click_polar_rotation.z)));
						m_camera->lookAtPoint(m_polar_rotation_origin, m_camera->rotation().z);

						if (m_animate)
						{
							m_rotation_vector = m_camera->position() - m_rotation_center;
							m_rotation_angle = 0.0f;
						}
					}
				}

				// focal point
				else if (ImGui::IsMouseClicked(ImGuiMouseButton_Right || !m_was_focused))
				{
					m_camera->focus(Math::vec2ui32(
						std::max(m_image_click_pos.x, 0),
						std::max(m_image_click_pos.y, 0)));
				}
				else
				{
					m_mouse_dragging = false;
				}

				// zoom
				if (const auto wheel = ImGui::GetIO().MouseWheel; wheel != 0.0f)
				{
					auto OC = m_camera->position() - m_polar_rotation_origin;
					const float step = 100.0f / zoom_speed;
					OC *= (step - std::min(wheel, step * 0.5f)) / step;
					m_camera->position(m_polar_rotation_origin + OC);

					if (m_animate)
					{
						m_rotation_vector = m_camera->position() - m_rotation_center;
						m_rotation_angle = 0.0f;
					}
				}

				m_was_focused = true;
			}
			else
			{
				m_was_focused = false;
			}
		}

		// animation
		if (m_animate)
		{
			auto rotated = m_rotation_vector.RotatedY(m_rotation_angle += m_rotation_speed * dt);
			m_camera->position(m_rotation_center + rotated);
			m_camera->lookAtPoint(m_rotation_center, m_camera->rotation().z);
		}
	}
	void Viewport::controlCanvas()
	{
		const auto content_center_pos = (m_content_max - m_content_min) / 2;
		if (m_reset_canvas)
		{
			m_zoom = 1.0f;
			m_canvas_center_pos = m_canvas_center_click_pos = Math::vec2f32{};
			m_reset_canvas = false;
		}

		if (ImGui::IsKeyDown(ImGuiKey_ModCtrl))
		{
			if (ImGui::IsMouseClicked(ImGuiMouseButton_Left))
			{
				m_content_mouse_click_pos = m_content_mouse_pos;
				m_canvas_center_click_pos = m_canvas_center_pos;
			}
			else if (ImGui::IsMouseDown(ImGuiMouseButton_Left))
			{
				m_canvas_center_pos = m_canvas_center_click_pos +
					Math::vec2f32(m_content_mouse_pos - m_content_mouse_click_pos);
			}

			const auto canvas_center_content_pos = Math::vec2f32(content_center_pos) + m_canvas_center_pos;
			const auto mouse_to_pos = (canvas_center_content_pos - Math::vec2f32(m_content_mouse_pos));

			if (const auto wheel_delta = ImGui::GetIO().MouseWheel; wheel_delta != 0.0f)
			{
				const float new_zoom = std::clamp(m_zoom + wheel_delta * m_zoom * 0.1f, 0.5f, 32.0f);
				const float zoom_factor = new_zoom / m_zoom;
				m_zoom = new_zoom;

				m_canvas_center_pos += mouse_to_pos * zoom_factor - mouse_to_pos;
			}
		}
	}
	void Viewport::drawRender()
	{
		if (!m_image.textureHandle()) return;

		const Math::vec2f32 image_res(float(m_image.width()), float(m_image.height()));
		const float scale_factor = std::min(m_content_res.x / image_res.x, m_content_res.y / image_res.y);
		const auto adjusted_res = image_res * scale_factor;
		const auto zoomed_res = adjusted_res * m_zoom;

		const Math::vec2f32 content_res(m_content_res);

		m_canvas_center_pos.x = std::clamp(
			m_canvas_center_pos.x,
			-zoomed_res.x / 2, zoomed_res.x / 2);
		m_canvas_center_pos.y = std::clamp(
			m_canvas_center_pos.y,
			-zoomed_res.y / 2, zoomed_res.y / 2);

		// draw render image
		const auto content_center_pos = Math::vec2f32(m_content_max - m_content_min) / 2.0f;
		const auto cursor_pos = content_center_pos + m_canvas_center_pos - (adjusted_res / 2) * m_zoom;
		ImGui::SetCursorScreenPos(toImVec2(Math::vec2f32(m_content_min) + cursor_pos));
		ImGui::Image(
			m_image.textureHandle(),
			toImVec2(zoomed_res));

		// update on image mouse click position
		const auto on_image_factor = image_res / Math::vec2f32(m_content_res);
		m_image_click_pos =
			(Math::vec2f32(m_content_mouse_pos) - cursor_pos) *
			(Math::vec2f32(m_content_res) / zoomed_res) * on_image_factor;

	}
	void Viewport::drawStats()
	{
		if (!m_stats) return;
		if (!m_camera) return;

		const float elapsed_time = m_stats->timer.time().count();
		m_stats->ft = m_stats->ft + (elapsed_time - m_stats->ft) * 0.1f;
		const float rps = [&]()
		{
			const auto prev_count = m_stats->prev_ray_count;
			const auto curr_count = m_camera->rayCount();
			const auto fps = 1000.0f / m_stats->ft;
			m_stats->prev_ray_count = curr_count;
			if (prev_count >= curr_count)
			{
				if (m_camera->enabled())
					return curr_count * fps;
				else
					return 0.0f;
			}
			return (curr_count - prev_count) * fps;
		}();

		bool open = true;
		if (ImGui::Begin("rendering", &open))
		{
			ImGui::Text("Traced rays: %s (%sr/s)",
				Utils::scientificWithPrefix(m_camera->rayCount()).c_str(),
				Utils::scientificWithPrefix(size_t(rps)).c_str());
		}
		ImGui::End();

		if (!open)
			m_stats.reset();
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
			auto [element, inserted] = m_viewports.insert(std::make_pair(id, Viewport(mr_world, std::move(camera), id + 1)));
			RZAssertCore(inserted, "failed to insert new Viewport");
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

	void Viewports::draw(const Rendering::Vulkan::Handle<VkCommandBuffer>& command_buffer)
	{
		for (auto& [id, viewport] : m_viewports)
			viewport.draw(command_buffer);
	}
	RZ::Handle<RZ::Camera> Viewports::getSelected()
	{
		for (auto& [id, viewport] : m_viewports)
			if (viewport.clicked())
				return viewport.camera();
		return {};
	}

	RZ::Handle<RZ::Mesh> Viewports::getSelectedMesh()
	{
		for (auto& [id, viewport] : m_viewports)
		{
			if (viewport.selected())
				return viewport.getSelectedMesh();
		}
		return {};
	}
	RZ::Handle<RZ::Material> Viewports::getSelectedMaterial()
	{
		return {};
	}
}
