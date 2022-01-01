#include "direct_light.h"
#include <algorithm>
#include "constants.h"

namespace RayZath::Engine
{
	DirectLight::DirectLight(
		Updatable* updatable,
		const ConStruct<DirectLight>& conStruct)
		: WorldObject(updatable, conStruct)
	{
		SetDirection(conStruct.direction);
		SetColor(conStruct.color);
		SetEmission(conStruct.emission);
		SetAngularSize(conStruct.angular_size);
	}
	DirectLight::~DirectLight()
	{}


	void DirectLight::SetDirection(const Math::vec3f& direction)
	{
		m_direction = direction;
		m_direction.Normalize();
		GetStateRegister().RequestUpdate();
	}
	void DirectLight::SetColor(const Graphics::Color& color)
	{
		m_color = color;
		GetStateRegister().RequestUpdate();
	}
	void DirectLight::SetEmission(const float& emission)
	{
		m_emission = std::max(emission, 0.0f);
		GetStateRegister().RequestUpdate();
	}
	void DirectLight::SetAngularSize(const float& angular_size)
	{
		m_angular_size = std::clamp(angular_size, 0.0f, Math::constants<float>::pi);
		GetStateRegister().RequestUpdate();
	}

	const Math::vec3f DirectLight::GetDirection() const noexcept
	{
		return m_direction;
	}
	const Graphics::Color DirectLight::GetColor() const noexcept
	{
		return m_color;
	}
	float DirectLight::GetEmission() const noexcept
	{
		return m_emission;
	}
	float DirectLight::GetAngularSize() const noexcept
	{
		return m_angular_size;
	}
}