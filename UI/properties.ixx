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

	export class Properties
		: public CameraProperties
		, public SpotLightProperties
		, public DirectLightProperties
	{
	private:
		std::variant<
			std::monostate,
			RZ::Handle<RZ::Camera>,
			RZ::Handle<RZ::SpotLight>,
			RZ::Handle<RZ::DirectLight>> m_type;
	public:
		template <typename T>
		void setObject(RZ::Handle<T> object) { m_type = std::move(object); }
		void displayCurrentObject();
	private:
		void displayEmpty();
	};
}
