#ifndef SPOT_LIGHT_H
#define SPOT_LIGHT_H

#include "world_object.h"
#include "vec3.h"
#include "color.h"

namespace RayZath
{
	class SpotLight;
	template<> struct ConStruct<SpotLight>;

	class SpotLight : public WorldObject
	{
	private:
		Math::vec3<float> m_position;
		Math::vec3<float> m_direction;
		Graphics::Color m_color;
		float m_size;
		float m_emission;

		float m_angle;
		float m_sharpness;


	public:
		SpotLight(
			Updatable* updatable,
			const ConStruct<SpotLight>& conStruct);
		~SpotLight();


	public:
		void SetPosition(const Math::vec3<float>& position);
		void SetDirection(const Math::vec3<float>& direction);
		void SetColor(const Graphics::Color& color);
		void SetSize(const float& size);
		void SetEmission(const float& emission);
		void SetBeamAngle(const float& angle);
		void SetSharpness(const float& sharpness);

		const Math::vec3<float>& GetPosition() const noexcept;
		const Math::vec3<float>& GetDirection() const noexcept;
		const Graphics::Color& GetColor() const noexcept;
		float GetSize() const noexcept;
		float GetEmission() const noexcept;
		float GetBeamAngle() const noexcept;
		float GetSharpness() const noexcept;
	};

	template<> struct ConStruct<SpotLight> : public ConStruct<WorldObject>
	{
		Math::vec3<float> position;
		Math::vec3<float> direction;
		Graphics::Color color;
		float size, emission;
		float beam_angle;
		float sharpness;

		ConStruct(
			const std::wstring& name = L"name",
			Math::vec3<float> position = Math::vec3<float>(0.0f, 5.0f, 0.0f),
			Math::vec3<float> direction = Math::vec3<float>(0.0f, -1.0f, 0.0f),
			Graphics::Color color = Graphics::Color(0xFF, 0xFF, 0xFF),
			float size = 0.5f,
			float emission = 100.0f,
			float beam_angle = 1.0f,
			float sharpness = 2.0f)
			: ConStruct<WorldObject>(name)
			, position(position)
			, direction(direction)
			, color(color)
			, size(size)
			, emission(emission)
			, beam_angle(beam_angle)
			, sharpness(sharpness)
		{}
	};
}

#endif // !SPOT_LIGHT_H