#include "spot_light.hpp"
#include <algorithm>

namespace RayZath::Engine
{
	SpotLight::SpotLight(
		Updatable* updatable,
		const ConStruct<SpotLight>& conStruct)
		: WorldObject(updatable, conStruct)
	{
		position(conStruct.position);
		direction(conStruct.direction);
		color(conStruct.color);
		SetSize(conStruct.size);
		emission(conStruct.emission);
		SetBeamAngle(conStruct.beam_angle);
	}

	void SpotLight::position(const Math::vec3f& position)
	{
		m_position = position;
		stateRegister().RequestUpdate();
	}
	void SpotLight::direction(const Math::vec3f& direction)
	{
		m_direction = direction;
		m_direction.Normalize();
		stateRegister().RequestUpdate();
	}
	void SpotLight::color(const Graphics::Color& color)
	{
		m_color = color;
		stateRegister().RequestUpdate();
	}
	void SpotLight::SetSize(const float& size)
	{
		m_size = std::max(size, std::numeric_limits<float>::min());
		stateRegister().RequestUpdate();
	}
	void SpotLight::emission(const float& emission)
	{
		m_emission = std::max(emission, 0.0f);
		stateRegister().RequestUpdate();
	}
	void SpotLight::SetBeamAngle(const float& angle)
	{
		m_angle = std::clamp(angle, 0.0f, 3.14159f);
		stateRegister().RequestUpdate();
	}

	const Math::vec3f& SpotLight::position() const noexcept
	{
		return m_position;
	}
	const Math::vec3f& SpotLight::direction() const noexcept
	{
		return m_direction;
	}
	const Graphics::Color& SpotLight::color() const noexcept
	{
		return m_color;
	}
	float SpotLight::size() const noexcept
	{
		return m_size;
	}
	float SpotLight::emission() const noexcept
	{
		return m_emission;
	}
	float SpotLight::GetBeamAngle() const noexcept
	{
		return m_angle;
	}
}