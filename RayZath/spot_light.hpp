#ifndef SPOT_LIGHT_H
#define SPOT_LIGHT_H

#include "world_object.hpp"

#include "vec3.h"
#include "color.h"

namespace RayZath::Engine
{
	class SpotLight;
	template<> struct ConStruct<SpotLight>;

	class SpotLight : public WorldObject
	{
	private:
		Math::vec3f m_position;
		Math::vec3f m_direction;
		Graphics::Color m_color;
		float m_size;
		float m_emission;
		float m_angle;


	public:
		SpotLight(
			Updatable* updatable,
			const ConStruct<SpotLight>& conStruct);
		~SpotLight();


	public:
		void position(const Math::vec3f& position);
		void direction(const Math::vec3f& direction);
		void color(const Graphics::Color& color);
		void SetSize(const float& size);
		void emission(const float& emission);
		void SetBeamAngle(const float& angle);

		const Math::vec3f& position() const noexcept;
		const Math::vec3f& direction() const noexcept;
		const Graphics::Color& color() const noexcept;
		float size() const noexcept;
		float emission() const noexcept;
		float GetBeamAngle() const noexcept;
	};

	template<> struct ConStruct<SpotLight> : public ConStruct<WorldObject>
	{
		Math::vec3f position;
		Math::vec3f direction;
		Graphics::Color color;
		float size = 0.5f, emission = 100.0f;
		float beam_angle = 1.0f;

		ConStruct(
			const std::string& name = "name",
			Math::vec3f position = Math::vec3f(0.0f, 5.0f, 0.0f),
			Math::vec3f direction = Math::vec3f(0.0f, -1.0f, 0.0f),
			Graphics::Color color = Graphics::Color(0xFF, 0xFF, 0xFF),
			float size = 0.5f,
			float emission = 100.0f,
			float beam_angle = 1.0f)
			: ConStruct<WorldObject>(name)
			, position(position)
			, direction(direction)
			, color(color)
			, size(size)
			, emission(emission)
			, beam_angle(beam_angle)
		{}
		explicit ConStruct(const Handle<SpotLight>& light)
		{
			if (!light) return;

			name = light->name();
			position = light->position();
			direction = light->direction();
			color = light->color();
			size = light->size();
			emission = light->emission();
			beam_angle = light->GetBeamAngle();
		}
	};
}

#endif // !SPOT_LIGHT_H