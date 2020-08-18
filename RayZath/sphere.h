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
		float m_radious;
		Graphics::Color m_color;
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
		void SetRadious(const float& radious);
		void SetColor(const Graphics::Color& color);

		float GetRadious() const noexcept;
		Graphics::Color& GetColor() noexcept;
		const Graphics::Color& GetColor() const noexcept;
		const Texture* GetTexture() const;


	public:
		friend class World;
	};

	template<> struct ConStruct<Sphere> : public ConStruct<RenderObject>
	{
		float radious;
		Graphics::Color color;

		ConStruct(
			const ConStruct<RenderObject>& renderObjectConStruct = ConStruct<RenderObject>(),
			float radious = 1.0f,
			const Graphics::Color& color = Graphics::Color(0xFF, 0xFF, 0xFF))
			: ConStruct<RenderObject>(renderObjectConStruct)
			, radious(radious)
			, color(color)
		{}
		~ConStruct()
		{}
	};
}

#endif // !SPHERE_H