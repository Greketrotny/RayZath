#include "point_light.h"

namespace RayZath::Engine
{
	PointLight::PointLight(
		Updatable* updatable,
		const ConStruct<PointLight>& conStruct)
		: WorldObject(updatable, conStruct)
	{
		SetPosition(conStruct.position);
		SetColor(conStruct.color);
		SetSize(conStruct.size);
		SetEmission(conStruct.emission);
	}
	PointLight::~PointLight()
	{
	}


	void PointLight::SetPosition(const Math::vec3f& position)
	{
		m_position = position;
		GetStateRegister().RequestUpdate();
	}
	void PointLight::SetColor(const Graphics::Color& color)
	{
		m_color = color;
		GetStateRegister().RequestUpdate();
	}
	void PointLight::SetSize(const float& size)
	{
		m_size = std::max(size, std::numeric_limits<float>::epsilon());
		GetStateRegister().RequestUpdate();
	}
	void PointLight::SetEmission(const float& emission)
	{
		m_emission = std::max(emission, std::numeric_limits<float>::epsilon());
		GetStateRegister().RequestUpdate();
	}

	const Math::vec3f& PointLight::GetPosition() const
	{
		return m_position;
	}
	const Graphics::Color& PointLight::GetColor() const
	{
		return m_color;
	}
	const float& PointLight::GetSize() const
	{
		return m_size;
	}
	const float& PointLight::GetEmission() const
	{
		return m_emission;
	}
}