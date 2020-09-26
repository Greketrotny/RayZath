#include "point_light.h"

namespace RayZath
{
	PointLight::PointLight(
		const size_t& id,
		Updatable* updatable,
		const ConStruct<PointLight>& conStruct)
		: WorldObject(id, updatable, conStruct)
	{
		SetPosition(conStruct.position);
		SetColor(conStruct.color);
		SetSize(conStruct.size);
		SetEmission(conStruct.emission);
	}
	PointLight::~PointLight()
	{
	}


	void PointLight::SetPosition(const Math::vec3<float>& position)
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

	const Math::vec3<float>& PointLight::GetPosition() const
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