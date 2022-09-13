#include "direct_light.hpp"
#include "constants.h"

#include <algorithm>

namespace RayZath::Engine
{
	DirectLight::DirectLight(
		Updatable* updatable,
		const ConStruct<DirectLight>& conStruct)
		: WorldObject(updatable, conStruct)
	{
		direction(conStruct.direction);
		color(conStruct.color);
		emission(conStruct.emission);
		SetAngularSize(conStruct.angular_size);
	}
	DirectLight::~DirectLight()
	{}


	void DirectLight::direction(const Math::vec3f& direction)
	{
		m_direction = direction;
		m_direction.Normalize();
		stateRegister().RequestUpdate();
	}
	void DirectLight::color(const Graphics::Color& color)
	{
		m_color = color;
		stateRegister().RequestUpdate();
	}
	void DirectLight::emission(const float& emission)
	{
		m_emission = std::max(emission, 0.0f);
		stateRegister().RequestUpdate();
	}
	void DirectLight::SetAngularSize(const float& angular_size)
	{
		m_angular_size = std::clamp(angular_size, 0.0f, Math::constants<float>::pi);
		stateRegister().RequestUpdate();
	}

	const Math::vec3f DirectLight::direction() const noexcept
	{
		return m_direction;
	}
	const Graphics::Color DirectLight::color() const noexcept
	{
		return m_color;
	}
	float DirectLight::emission() const noexcept
	{
		return m_emission;
	}
	float DirectLight::angularSize() const noexcept
	{
		return m_angular_size;
	}
}