#ifndef POINT_LIGHT_H
#define POINT_LIGHT_H

#include "world_object.h"

#include "vec3.h"
#include "Color.h"

namespace RayZath::Engine
{
	class PointLight;
	template<> struct ConStruct<PointLight>;

	class PointLight : public WorldObject
	{
	private:
		Math::vec3f m_position;
		Graphics::Color m_color;
		float m_size;
		float m_emission;


	public:
		PointLight(
			Updatable* updatable,
			const ConStruct<PointLight>& con_struct);
		~PointLight();


	public:
		void SetPosition(const Math::vec3f& position);
		void SetColor(const Graphics::Color& color);
		void SetSize(const float& size);
		void SetEmission(const float& emission);

		const Math::vec3f& GetPosition() const;
		const Graphics::Color& GetColor() const;
		const float& GetSize() const;
		const float& GetEmission() const;
	};

	template<> struct ConStruct<PointLight> : public ConStruct<WorldObject>
	{
		Math::vec3f position;
		Graphics::Color color;
		float size;
		float emission;

		ConStruct(
			const std::string& name = "name",
			Math::vec3f position = Math::vec3f(0.0f, 5.0f, 0.0f),
			Graphics::Color color = Graphics::Color(0xFF, 0xFF, 0xFF),
			float size = 0.5f,
			float emission = 100.0f)
			: ConStruct<WorldObject>(name)
			, position(position)
			, color(color)
			, size(size)
			, emission(emission)
		{}
	};
}

#endif // !LIGHT_H