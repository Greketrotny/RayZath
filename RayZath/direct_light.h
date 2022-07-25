#ifndef DIRECT_LIGHT_H
#define DIRECT_LIGHT_H

#include "world_object.h"
#include "vec3.h"
#include "color.h"

namespace RayZath::Engine
{
	class DirectLight;
	template<> struct ConStruct<DirectLight>;

	class DirectLight : public WorldObject
	{
	private:
		Math::vec3f m_direction;
		Graphics::Color m_color;
		float m_emission;
		float m_angular_size;


	public:
		DirectLight(
			Updatable* updatable,
			const ConStruct<DirectLight>& conStruct);
		~DirectLight();


	public:
		void SetDirection(const Math::vec3f& direction);
		void SetColor(const Graphics::Color& color);
		void SetEmission(const float& emission);
		void SetAngularSize(const float& angular_size);

		const Math::vec3f GetDirection() const noexcept;
		const Graphics::Color GetColor() const noexcept;
		float GetEmission() const noexcept;
		float GetAngularSize() const noexcept;
	};

	template<> struct ConStruct<DirectLight> : public ConStruct<WorldObject>
	{
		Math::vec3f direction;
		Graphics::Color color;
		float emission = 100.0f;
		float angular_size = 0.1f;

		ConStruct(
			const std::string name = "name",
			Math::vec3f direction = Math::vec3f(0.0f, -1.0f, 0.0f),
			Graphics::Color color = Graphics::Color(0xFF, 0xFF, 0xFF),
			float emission = 100.0f,
			float angular_size = 0.1f)
			: ConStruct<WorldObject>(name)
			, direction(direction)
			, color(color)
			, emission(emission)
			, angular_size(angular_size)
		{}
		explicit ConStruct(const Handle<DirectLight>& light)
		{
			if (!light) return;

			name = light->GetName();
			direction = light->GetDirection();
			color = light->GetColor();
			emission = light->GetEmission();
			angular_size = light->GetAngularSize();
		}
	};
}

#endif // !DIRECT_LIGHT_H