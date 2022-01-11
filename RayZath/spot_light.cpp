#include "spot_light.h"
#include <algorithm>

namespace RayZath::Engine
{
	SpotLight::SpotLight(
		Updatable* updatable,
		const ConStruct<SpotLight>& conStruct)
		: WorldObject(updatable, conStruct)
	{
		SetPosition(conStruct.position);
		SetDirection(conStruct.direction);
		SetColor(conStruct.color);
		SetSize(conStruct.size);
		SetEmission(conStruct.emission);
		SetBeamAngle(conStruct.beam_angle);
	}
	SpotLight::~SpotLight()
	{
	}

	void SpotLight::SetPosition(const Math::vec3f& position)
	{
		m_position = position;
		GetStateRegister().RequestUpdate();
	}
	void SpotLight::SetDirection(const Math::vec3f& direction)
	{
		m_direction = direction;
		m_direction.Normalize();
		GetStateRegister().RequestUpdate();
	}
	void SpotLight::SetColor(const Graphics::Color& color)
	{
		m_color = color;
		GetStateRegister().RequestUpdate();
	}
	void SpotLight::SetSize(const float& size)
	{
		m_size = std::max(size, std::numeric_limits<float>::min());
		GetStateRegister().RequestUpdate();
	}
	void SpotLight::SetEmission(const float& emission)
	{
		m_emission = std::max(emission, 0.0f);
		GetStateRegister().RequestUpdate();
	}
	void SpotLight::SetBeamAngle(const float& angle)
	{
		m_angle = std::clamp(angle, 0.0f, 3.14159f);
		GetStateRegister().RequestUpdate();
	}

	const Math::vec3f& SpotLight::GetPosition() const noexcept
	{
		return m_position;
	}
	const Math::vec3f& SpotLight::GetDirection() const noexcept
	{
		return m_direction;
	}
	const Graphics::Color& SpotLight::GetColor() const noexcept
	{
		return m_color;
	}
	float SpotLight::GetSize() const noexcept
	{
		return m_size;
	}
	float SpotLight::GetEmission() const noexcept
	{
		return m_emission;
	}
	float SpotLight::GetBeamAngle() const noexcept
	{
		return m_angle;
	}
}