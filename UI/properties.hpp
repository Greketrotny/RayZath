#pragma once

#include "rayzath.h"

#include <variant>

namespace RayZath::UI::Windows
{
	namespace RZ = RayZath::Engine;
	using ObjectType = Engine::World::ObjectType;

	template <Engine::World::ObjectType T>
	class PropertiesBase
	{
	protected:
		RZ::Handle<Engine::World::object_t<T>> m_object;
		float m_label_width = 100.0f;

	public:
		PropertiesBase(float label_width = 100.0f)
			: m_label_width(label_width)
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
		using PropertiesBase<Engine::World::ObjectType::Material>::setObject;
		void setObject(Engine::Material* material);
		void display();
	private:
		void display(RZ::Material& material);
	};
	template<>
	class Properties<Engine::World::ObjectType::Camera> : public PropertiesBase<Engine::World::ObjectType::Camera>
	{
	public:
		Properties()
			: PropertiesBase(120.0f)
		{}

		void display();
	};
	template<>
	class Properties<Engine::World::ObjectType::SpotLight> : public PropertiesBase<Engine::World::ObjectType::SpotLight>
	{
	public:
		void display();
	};
	template<>
	class Properties<Engine::World::ObjectType::DirectLight> : public PropertiesBase<Engine::World::ObjectType::DirectLight>
	{
	public:
		void display();
	};
	template<>
	class Properties<Engine::World::ObjectType::Mesh> : public PropertiesBase<Engine::World::ObjectType::Mesh>
	{
		RZ::Handle<RZ::Material> m_selected_material;
		Properties<Engine::World::ObjectType::Material> m_material_properties;

	public:
		void display();
		void reset() { m_selected_material.Release(); }
	};
	template<>
	class Properties<Engine::World::ObjectType::Group> : public PropertiesBase<Engine::World::ObjectType::Group>
	{
	public:
		void display();
	};

	template<>
	class Properties<Engine::World::ObjectType::Texture> : public PropertiesBase<Engine::World::ObjectType::Texture>
	{
	public:
		void display();
	};
	template<>
	class Properties<Engine::World::ObjectType::NormalMap> : public PropertiesBase<Engine::World::ObjectType::NormalMap>
	{
	public:
		void display();
	};
	template<>
	class Properties<Engine::World::ObjectType::MetalnessMap> : public PropertiesBase<Engine::World::ObjectType::MetalnessMap>
	{
	public:
		void display();
	};
	template<>
	class Properties<Engine::World::ObjectType::RoughnessMap> : public PropertiesBase<Engine::World::ObjectType::RoughnessMap>
	{
	public:
		void display();
	};
	template<>
	class Properties<Engine::World::ObjectType::EmissionMap> : public PropertiesBase<Engine::World::ObjectType::EmissionMap>
	{
	public:
		void display();
	};

	class MultiProperties
	{
	private:
		std::variant<
			std::monostate,
			Properties<ObjectType::Camera>,
			Properties<ObjectType::SpotLight>,
			Properties<ObjectType::DirectLight>,
			Properties<ObjectType::Mesh>,
			Properties<ObjectType::Group>,
			Properties<ObjectType::Material>,

			Properties<ObjectType::Texture>,
			Properties<ObjectType::NormalMap>,
			Properties<ObjectType::MetalnessMap>,
			Properties<ObjectType::RoughnessMap>,
			Properties<ObjectType::EmissionMap>> m_type;
	public:
		template <Engine::World::ObjectType T>
		void setObject(RZ::Handle<Engine::World::object_t<T>> object)
		{
			if (!std::holds_alternative<Properties<T>>(m_type))
				m_type.emplace<Properties<T>>(Properties<T>{});

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
				m_type.emplace<Properties<T>>(Properties<T>{});

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
