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
		Texture* m_pTexture = nullptr;


	private:
		Sphere(const Sphere&) = delete;
		Sphere(Sphere&&) = delete;
		Sphere(
			const size_t& id, 
			Updatable* updatable,
			const ConStruct<Sphere>& conStruct);
		~Sphere();


	public:
		Sphere& operator=(const Sphere&) = delete;
		Sphere& operator=(Sphere&&) = delete;


	public:
		void LoadTexture(const Texture& newTexture);
		void UnloadTexture();
		void SetRadius(const float& radius);

		float GetRadius() const noexcept;
		const Texture* GetTexture() const;

		void Update() override;


	public:
		friend class ObjectCreator;
	};

	template<> struct ConStruct<Sphere> : public ConStruct<RenderObject>
	{
		float radius;

		ConStruct(
			const ConStruct<RenderObject>& renderObjectConStruct = ConStruct<RenderObject>(),
			float radius = 1.0f)
			: ConStruct<RenderObject>(renderObjectConStruct)
			, radius(radius)
		{}
		~ConStruct()
		{}
	};
}

#endif // !SPHERE_H