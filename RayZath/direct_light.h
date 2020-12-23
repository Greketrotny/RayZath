#ifndef DIRECT_LIGHT_H
#define DIRECT_LIGHT_H

#include "world_object.h"
#include "vec3.h"
#include "color.h"

namespace RayZath
{
	class DirectLight;
	template<> struct ConStruct<DirectLight>;

	class DirectLight : public WorldObject
	{
	private:
		Math::vec3<float> m_direction;
		Graphics::Color m_color;
		float m_emission;
		float m_angular_size;


	public:
		DirectLight(
			Updatable* updatable,
			const ConStruct<DirectLight>& conStruct);
		~DirectLight();


	public:
		void SetDirection(const Math::vec3<float>& direction);
		void SetColor(const Graphics::Color& color);
		void SetEmission(const float& emission);
		void SetAngularSize(const float& angular_size);

		const Math::vec3<float> GetDirection() const noexcept;
		const Graphics::Color GetColor() const noexcept;
		float GetEmission() const noexcept;
		float GetAngularSize() const noexcept;
	};


	template<> struct ConStruct<DirectLight> : public ConStruct<WorldObject>
	{
		Math::vec3<float> direction;
		Graphics::Color color;
		float emission;
		float angular_size;

		ConStruct(
			const std::wstring name = L"name",
			Math::vec3<float> direction = Math::vec3<float>(0.0f, -1.0f, 0.0f),
			Graphics::Color color = Graphics::Color(0xFF, 0xFF, 0xFF),
			float emission = 100.0f,
			float angular_size = 0.1f)
			: ConStruct<WorldObject>(name)
			, direction(direction)
			, color(color)
			, emission(emission)
			, angular_size(angular_size)
		{}
	};
}

#endif // !DIRECT_LIGHT_H