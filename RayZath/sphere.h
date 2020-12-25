#ifndef SPHERE_H
#define SPHERE_H

#include "render_object.h"
#include "render_parts.h"

#include "color.h"
#include <numeric>

namespace RayZath
{
	class Sphere;
	template<> struct ConStruct<Sphere>;

	class Sphere : public RenderObject
	{
	private:
		float m_radius;
		Observer<Material> m_material;


	public:
		Sphere(const Sphere&) = delete;
		Sphere(Sphere&&) = delete;
		Sphere(
			Updatable* updatable,
			const ConStruct<Sphere>& conStruct);
		~Sphere();


	public:
		Sphere& operator=(const Sphere&) = delete;
		Sphere& operator=(Sphere&&) = delete;


	public:
		void SetRadius(const float& radius);
		float GetRadius() const noexcept;

		void SetMaterial(const Handle<Material>& material);
		const Handle<Material>& GetMaterial() const;

		void Update() override;
		void NotifyMaterial();
	};

	template<> struct ConStruct<Sphere> : public ConStruct<RenderObject>
	{
		float radius;
		Handle<Material> material;

		ConStruct(
			const std::wstring& name = L"name",
			const Math::vec3<float>& position = Math::vec3<float>(0.0f, 0.0f, 0.0f),
			const Math::vec3<float>& rotation = Math::vec3<float>(0.0f, 0.0f, 0.0f),
			const Math::vec3<float>& center = Math::vec3<float>(0.0f, 0.0f, 0.0f),
			const Math::vec3<float>& scale = Math::vec3<float>(1.0f, 1.0f, 1.0f),
			const Handle<Material>& material = Handle<Material>(),
			float radius = 1.0f)
			: ConStruct<RenderObject>(
				name, position, rotation, center, scale)
			, radius(radius)
			, material(material)
		{}
		~ConStruct()
		{}
	};
}

#endif // !SPHERE_H