#ifndef PLANE_H
#define PLANE_H

#include "render_object.h"

namespace RayZath
{
	class Plane;
	template<> struct ConStruct<Plane>;

	class Plane
		: public RenderObject
	{
	private:
		Observer<Material> m_material;


	public:
		Plane(
			Updatable* updatable,
			const ConStruct<Plane>& con_struct);


		void SetMaterial(const Handle<Material>& material);
		const Handle<Material>& GetMaterial() const noexcept;
	private:
		void NotifyMaterial();
	};

	template<> struct ConStruct<Plane>
		: public ConStruct<RenderObject>
	{
		Handle<Material> material;

		ConStruct(
			const std::wstring& name = L"name",
			const Math::vec3<float>& position = Math::vec3<float>(0.0f, 0.0f, 0.0f),
			const Math::vec3<float>& rotation = Math::vec3<float>(0.0f, 0.0f, 0.0f),
			const Math::vec3<float>& center = Math::vec3<float>(0.0f, 0.0f, 0.0f),
			const Math::vec3<float>& scale = Math::vec3<float>(1.0f, 1.0f, 1.0f),
			const Handle<Material>& material = Handle<Material>())
			: ConStruct<RenderObject>(
				name, position, rotation, center, scale)
			, material(material)
		{}
	};
}

#endif // !PLANE_H