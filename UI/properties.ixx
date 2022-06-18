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
		void display(const RZ::Handle<RZ::Material>& material);
		void display(RZ::Material& material);
	};

	export class TextureProperties : public PropertiesBase
	{
	public:
		void display(const RZ::Handle<RZ::Texture>& texture);
	};
	export class NormalMapProperties : public PropertiesBase
	{
	public:
		void display(const RZ::Handle<RZ::NormalMap>& map);
	};
	export class MetalnessMapProperties : public PropertiesBase
	{
	public:
		void display(const RZ::Handle<RZ::MetalnessMap>& map);
	};
	export class RoughnessMapProperties : public PropertiesBase
	{
	public:
		void display(const RZ::Handle<RZ::RoughnessMap>& map);
	};
	export class EmissionMapProperties : public PropertiesBase
	{
	public:
		void display(const RZ::Handle<RZ::EmissionMap>& map);
	};

	export class Properties
		: public CameraProperties
		, public SpotLightProperties
		, public DirectLightProperties
		, public MeshProperties
		, public GroupProperties
		, public MaterialProperties

		, public TextureProperties
		, public NormalMapProperties
		, public MetalnessMapProperties
		, public RoughnessMapProperties
		, public EmissionMapProperties
	{
	private:
		std::variant<
			std::monostate,
			RZ::Handle<RZ::Camera>,
			RZ::Handle<RZ::SpotLight>,
			RZ::Handle<RZ::DirectLight>,
			RZ::Handle<RZ::Mesh>,
			RZ::Handle<RZ::Group>,
			RZ::Handle<RZ::Material>,

			RZ::Handle<RZ::Texture>,
			RZ::Handle<RZ::NormalMap>,
			RZ::Handle<RZ::MetalnessMap>,
			RZ::Handle<RZ::RoughnessMap>,
			RZ::Handle<RZ::EmissionMap>> m_type;
		RZ::Material* m_material;
	public:
		template <size_t idx, typename T>
		void setObject(RZ::Handle<T> object)
		{
			m_material = nullptr;
			if (m_type.index() == idx &&
				std::get<idx>(m_type) == object) return;

			m_type.emplace<idx>(std::move(object));
			reset<T>();
		}
		void setObject(RZ::Material& material)
		{
			m_material = &material;
		}
		void displayCurrentObject();
	private:
		void displayEmpty();

		template <typename T> void reset() {}
		template <> void reset<RZ::Mesh>() { MeshProperties::reset(); }
	};
}
