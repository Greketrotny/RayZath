module;

#include "rayzath.h"

#include <variant>

export module rz.ui.windows.properties;

namespace RZ = RayZath::Engine;

namespace RayZath::UI::Windows
{
	export class PropertiesBase
	{
	protected:
		float m_label_width = 100.0f;

	public:
		PropertiesBase(float label_width = 100.0f);
	};

	export class CameraProperties : public PropertiesBase
	{
	public:
		CameraProperties()
			: PropertiesBase(120.0f)
		{}

		void display(const RZ::Handle<RZ::Camera>& camera);
	};
	export class SpotLightProperties : public PropertiesBase
	{
	public:
		void display(const RZ::Handle<RZ::SpotLight>& light);
	};
	export class DirectLightProperties : public PropertiesBase
	{
	public:
		void display(const RZ::Handle<RZ::DirectLight>& light);
	};
	export class MeshProperties : public PropertiesBase
	{
		RZ::Handle<RZ::Material> m_selected_material;

	public:
		void display(const RZ::Handle<RZ::Mesh>& object);
		void reset() { m_selected_material.Release(); }
	};
	export class GroupProperties : public PropertiesBase
	{
	public:
		void display(const RZ::Handle<RZ::Group>& group);
	};
	export class MaterialProperties : public PropertiesBase
	{
	public:
		void display(const RZ::Handle<RZ::Material>& group);
	};

	export class Properties
		: public CameraProperties
		, public SpotLightProperties
		, public DirectLightProperties
		, public MeshProperties
		, public GroupProperties
		, public MaterialProperties
	{
	private:
		std::variant<
			std::monostate,
			RZ::Handle<RZ::Camera>,
			RZ::Handle<RZ::SpotLight>,
			RZ::Handle<RZ::DirectLight>,
			RZ::Handle<RZ::Mesh>,
			RZ::Handle<RZ::Group>,
			RZ::Handle<RZ::Material>> m_type;
	public:
		template <typename T>
		void setObject(RZ::Handle<T> object)
		{
			if (std::holds_alternative<RZ::Handle<T>>(m_type) &&
				std::get<RZ::Handle<T>>(m_type) == object) return;

			m_type = std::move(object);
			reset<T>();
		}
		void displayCurrentObject();
	private:
		void displayEmpty();

		template <typename T> void reset() {}
		template <> void reset<RZ::Mesh>() { MeshProperties::reset(); }
	};
}
