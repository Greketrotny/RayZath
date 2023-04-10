#pragma once

#include "explorer_base.hpp"
#include "rayzath.hpp"

#include "imgui.h"

#include <variant>

namespace RayZath::UI::Windows
{
	namespace RZ = RayZath::Engine;
	using ObjectType = RZ::ObjectType;

	template <Engine::ObjectType T>
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
	class DirectLightPropertiesBase
	{
	protected:
		std::reference_wrapper<RZ::World> mr_world;
		RZ::SR::Handle<Engine::DirectLight> m_object;
		float m_label_width = 100.0f;

	public:
		DirectLightPropertiesBase(std::reference_wrapper<RZ::World> r_world, float label_width = 100.0f)
			: mr_world(r_world)
			, m_label_width(label_width)
		{}

		const auto& getObject() const { return m_object; }
		void setObject(RZ::SR::Handle<Engine::DirectLight> object) { m_object = std::move(object); }
		void reset() {};
	};

	template <Engine::ObjectType T>
	class Properties;

	template<>
	class Properties<Engine::ObjectType::Material>
		: public PropertiesBase<Engine::ObjectType::Material>
		, public ExplorerSelectable
	{
	private:
		Engine::Material* mp_material = nullptr;

		std::variant<
			std::monostate,
			Search<ObjectType::Texture>,
			Search<ObjectType::NormalMap>,
			Search<ObjectType::MetalnessMap>,
			Search<ObjectType::RoughnessMap>,
			Search<ObjectType::EmissionMap>> m_search_modal;

	public:
		Properties(std::reference_wrapper<RZ::World> r_world);

		using PropertiesBase<Engine::ObjectType::Material>::setObject;
		void setObject(Engine::Material* material);
		void display();
	private:
		void display(RZ::Material& material);
	};
	template<>
	class Properties<Engine::ObjectType::Mesh>
		: public PropertiesBase<Engine::ObjectType::Mesh>
	{
		using PropertiesBase<Engine::ObjectType::Mesh>::m_object;
		Engine::Transformation m_transformation;

	public:
		Properties(std::reference_wrapper<RZ::World> r_world);

		void display();
	};

	template<>
	class Properties<Engine::ObjectType::Camera> : public PropertiesBase<Engine::ObjectType::Camera>
	{
	public:
		Properties(std::reference_wrapper<RZ::World> r_world);

		void display();
	};
	template<>
	class Properties<Engine::ObjectType::SpotLight> : public PropertiesBase<Engine::ObjectType::SpotLight>
	{
	public:
		Properties(std::reference_wrapper<RZ::World> r_world);

		void display();
	};
	template<>
	class Properties<Engine::ObjectType::DirectLight> : public DirectLightPropertiesBase
	{
	public:
		Properties(std::reference_wrapper<RZ::World> r_world);

		void display();
	};
	template<>
	class Properties<Engine::ObjectType::Instance>
		: public PropertiesBase<Engine::ObjectType::Instance>
		, public ExplorerSelectable
	{
		RZ::Handle<RZ::Material> m_selected_material;
		RZ::Handle<RZ::Mesh> m_selected_mesh;
		Properties<Engine::ObjectType::Material> m_material_properties;

		std::unique_ptr<Search<ObjectType::Material>> m_search_modal;
		uint32_t m_selected_material_idx = 0;

	public:
		Properties(std::reference_wrapper<RZ::World> r_world);

		void display();
		void reset();
	};
	template<>
	class Properties<Engine::ObjectType::Group> : public PropertiesBase<Engine::ObjectType::Group>
	{
	public:
		Properties(std::reference_wrapper<RZ::World> r_world);

		void display();
	};

	template<>
	class Properties<Engine::ObjectType::Texture> : public PropertiesBase<Engine::ObjectType::Texture>
	{
	public:
		Properties(std::reference_wrapper<RZ::World> r_world);

		void display();
	};
	template<>
	class Properties<Engine::ObjectType::NormalMap> : public PropertiesBase<Engine::ObjectType::NormalMap>
	{
	public:
		Properties(std::reference_wrapper<RZ::World> r_world);

		void display();
	};
	template<>
	class Properties<Engine::ObjectType::MetalnessMap> : public PropertiesBase<Engine::ObjectType::MetalnessMap>
	{
	public:
		Properties(std::reference_wrapper<RZ::World> r_world);

		void display();
	};
	template<>
	class Properties<Engine::ObjectType::RoughnessMap> : public PropertiesBase<Engine::ObjectType::RoughnessMap>
	{
	public:
		Properties(std::reference_wrapper<RZ::World> r_world);

		void display();
	};
	template<>
	class Properties<Engine::ObjectType::EmissionMap> : public PropertiesBase<Engine::ObjectType::EmissionMap>
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
			Properties<ObjectType::Instance>,
			Properties<ObjectType::Group>,
			Properties<ObjectType::Material>,
			Properties<ObjectType::Mesh>,

			Properties<ObjectType::Texture>,
			Properties<ObjectType::NormalMap>,
			Properties<ObjectType::MetalnessMap>,
			Properties<ObjectType::RoughnessMap>,
			Properties<ObjectType::EmissionMap>> m_type;
	public:
		MultiProperties(std::reference_wrapper<RZ::World> world)
			: mr_world(world)
		{}

		template <Engine::ObjectType T>
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
		template <Engine::ObjectType T>
		void setObject(Engine::World::object_t<T>* object)
		{
			if (!std::holds_alternative<Properties<T>>(m_type))
				m_type.emplace<Properties<T>>(Properties<T>{mr_world});

			auto& props = std::get<Properties<T>>(m_type);
			props.setObject(object);
			props.reset();
		}
		template <Engine::ObjectType T>
		auto& getProperties() { return std::get<Properties<T>>(m_type); }
		void displayCurrentObject();
	private:
		void displayEmpty();
	};
}
