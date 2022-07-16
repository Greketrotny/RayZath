#pragma once

#include "rayzath.h"
#include "explorer_base.hpp"

#include "imgui.h"

#include <variant>

namespace RayZath::UI::Windows
{
	namespace RZ = RayZath::Engine;
	using ObjectType = RZ::World::ObjectType;

	template <ObjectType T>
	class Search
	{
	public:
		using object_t = RZ::World::object_t<T>;
	private:
		bool m_opened = true;
		std::array<char, 256> m_buffer{};
		bool m_match_case = false, m_match_word = false;

		RZ::Handle<object_t> m_selected;

	private:
		bool matches(const std::string& name)
		{
			std::string search_input(m_buffer.data());

			static_assert(std::tuple_size_v<decltype(m_buffer)> > 0);
			if (m_buffer[0] == '\0') return true; // no input, matches all

			if (m_match_case)
			{
				return name.find(search_input) != std::string::npos;
			}
			else
			{
				std::transform(search_input.begin(), search_input.end(), search_input.begin(),
					[](auto c) { return std::tolower(c); });
				std::string lowered_name = name;
				std::transform(lowered_name.begin(), lowered_name.end(), lowered_name.begin(),
					[](auto c) { return std::tolower(c); });
				return lowered_name.find(search_input) != std::string::npos;
			}
		}
	public:
		std::tuple<bool, RZ::Handle<object_t>> update(const RZ::World& world)
		{
			static constexpr auto* popup_id = "search##search_popup_modal";
			ImGui::OpenPopup(popup_id);
			if (ImGui::BeginPopupModal(popup_id, &m_opened))
			{
				ImGui::SetNextItemWidth(-1);
				ImGui::InputTextWithHint("##object_name_input", "name",
					m_buffer.data(), m_buffer.size(),
					ImGuiInputTextFlags_AlwaysOverwrite);
				ImGui::Checkbox("match case", &m_match_case);

				ImGui::Separator();

				if (ImGui::BeginListBox("search_list_box", ImVec2(-1.f, -1.f)))
				{
					auto& objects = world.Container<T>();
					for (uint32_t i = 0; i < objects.GetCount(); i++)
					{
						const auto& object = objects[i];
						if (!object) continue;
						if (!matches(object->GetName())) continue;

						bool is_selected = m_selected == object;
						if (ImGui::Selectable(object->GetName().c_str(), &is_selected))
							m_selected = object;

						if (is_selected)
							ImGui::SetItemDefaultFocus();
					}
					ImGui::EndListBox();
				}
				ImGui::EndPopup();
			}

			return std::make_tuple(m_opened, m_selected);
		}
	};

	template <Engine::World::ObjectType T>
	class PropertiesBase
	{
	protected:
		std::reference_wrapper<RZ::World> mr_world;
		RZ::Handle<Engine::World::object_t<T>> m_object;
		float m_label_width = 100.0f;

	public:
		PropertiesBase(std::reference_wrapper<RZ::World> r_world, float label_width = 100.0f)
			: mr_world(r_world)
			, m_label_width(label_width)
		{}

		const auto& getObject() const { return m_object; }
		void setObject(RZ::Handle<Engine::World::object_t<T>> object) { m_object = std::move(object); }
		void reset() {};
	};

	template <Engine::World::ObjectType T>
	class Properties;

	template<>
	class Properties<Engine::World::ObjectType::Material> : public PropertiesBase<Engine::World::ObjectType::Material>
	{
	private:
		Engine::Material* mp_material = nullptr;

	public:
		Properties(std::reference_wrapper<RZ::World> r_world);

		using PropertiesBase<Engine::World::ObjectType::Material>::setObject;
		void setObject(Engine::Material* material);
		void display();
	private:
		void display(RZ::Material& material);
	};
	template<>
	class Properties<Engine::World::ObjectType::MeshStructure> 
		: public PropertiesBase<Engine::World::ObjectType::MeshStructure>
	{
		using PropertiesBase<Engine::World::ObjectType::MeshStructure>::m_object;
	public:
		Properties(std::reference_wrapper<RZ::World> r_world);

		void display();
	};

	template<>
	class Properties<Engine::World::ObjectType::Camera> : public PropertiesBase<Engine::World::ObjectType::Camera>
	{
	public:
		Properties(std::reference_wrapper<RZ::World> r_world);

		void display();
	};
	template<>
	class Properties<Engine::World::ObjectType::SpotLight> : public PropertiesBase<Engine::World::ObjectType::SpotLight>
	{
	public:
		Properties(std::reference_wrapper<RZ::World> r_world);

		void display();
	};
	template<>
	class Properties<Engine::World::ObjectType::DirectLight> : public PropertiesBase<Engine::World::ObjectType::DirectLight>
	{
	public:
		Properties(std::reference_wrapper<RZ::World> r_world);

		void display();
	};
	template<>
	class Properties<Engine::World::ObjectType::Mesh> 
		: public PropertiesBase<Engine::World::ObjectType::Mesh>
		, public ExplorerSelectable
	{
		RZ::Handle<RZ::Material> m_selected_material;
		RZ::Handle<RZ::MeshStructure> m_selected_mesh;
		Properties<Engine::World::ObjectType::Material> m_material_properties;

		std::unique_ptr<Search<ObjectType::Material>> m_search_modal;
		uint32_t m_selected_material_idx = 0;

	public:
		Properties(std::reference_wrapper<RZ::World> r_world);

		void display();
		void reset();
	};
	template<>
	class Properties<Engine::World::ObjectType::Group> : public PropertiesBase<Engine::World::ObjectType::Group>
	{
	public:
		Properties(std::reference_wrapper<RZ::World> r_world);

		void display();
	};

	template<>
	class Properties<Engine::World::ObjectType::Texture> : public PropertiesBase<Engine::World::ObjectType::Texture>
	{
	public:
		Properties(std::reference_wrapper<RZ::World> r_world);

		void display();
	};
	template<>
	class Properties<Engine::World::ObjectType::NormalMap> : public PropertiesBase<Engine::World::ObjectType::NormalMap>
	{
	public:
		Properties(std::reference_wrapper<RZ::World> r_world);

		void display();
	};
	template<>
	class Properties<Engine::World::ObjectType::MetalnessMap> : public PropertiesBase<Engine::World::ObjectType::MetalnessMap>
	{
	public:
		Properties(std::reference_wrapper<RZ::World> r_world);

		void display();
	};
	template<>
	class Properties<Engine::World::ObjectType::RoughnessMap> : public PropertiesBase<Engine::World::ObjectType::RoughnessMap>
	{
	public:
		Properties(std::reference_wrapper<RZ::World> r_world);

		void display();
	};
	template<>
	class Properties<Engine::World::ObjectType::EmissionMap> : public PropertiesBase<Engine::World::ObjectType::EmissionMap>
	{
	public:
		Properties(std::reference_wrapper<RZ::World> r_world);

		void display();
	};

	class MultiProperties
	{
	private:
		std::reference_wrapper<RZ::World> mr_world;
		std::variant<
			std::monostate,
			Properties<ObjectType::Camera>,
			Properties<ObjectType::SpotLight>,
			Properties<ObjectType::DirectLight>,
			Properties<ObjectType::Mesh>,
			Properties<ObjectType::Group>,
			Properties<ObjectType::Material>,
			Properties<ObjectType::MeshStructure>,

			Properties<ObjectType::Texture>,
			Properties<ObjectType::NormalMap>,
			Properties<ObjectType::MetalnessMap>,
			Properties<ObjectType::RoughnessMap>,
			Properties<ObjectType::EmissionMap>> m_type;
	public:
		MultiProperties(std::reference_wrapper<RZ::World> world)
			: mr_world(world)
		{}

		template <Engine::World::ObjectType T>
		void setObject(RZ::Handle<Engine::World::object_t<T>> object)
		{
			if (!std::holds_alternative<Properties<T>>(m_type))
				m_type.emplace<Properties<T>>(Properties<T>{mr_world});

			if (std::get<Properties<T>>(m_type).getObject() != object)
			{
				auto& props = std::get<Properties<T>>(m_type);
				props.setObject(std::move(object));
				props.reset();
			}
		}
		template <Engine::World::ObjectType T>
		void setObject(Engine::World::object_t<T>* object)
		{
			if (!std::holds_alternative<Properties<T>>(m_type))
				m_type.emplace<Properties<T>>(Properties<T>{mr_world});

			auto& props = std::get<Properties<T>>(m_type);
			props.setObject(object);
			props.reset();
		}
		template <Engine::World::ObjectType T>
		auto& getProperties() { return std::get<Properties<T>>(m_type); }
		void displayCurrentObject();
	private:
		void displayEmpty();
	};
}
