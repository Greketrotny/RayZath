#include "Sphere.h"

namespace RayZath
{
	Sphere::Sphere(
		const size_t& id,
		Updatable* updatable,
		const ConStruct<Sphere>& conStruct)
		: RenderObject(id, updatable, conStruct)
	{
		SetRadious(conStruct.radious);
		SetColor(conStruct.color);
	}
	Sphere::~Sphere()
	{
		UnloadTexture();
	}

	void Sphere::LoadTexture(const Texture& newTexture)
	{
		if (m_pTexture == nullptr)	m_pTexture = new Texture(newTexture);
		else *m_pTexture = newTexture;
		RequestUpdate();
	}
	void Sphere::UnloadTexture()
	{
		if (m_pTexture) delete m_pTexture;
		m_pTexture = nullptr;
		RequestUpdate();
	}
	void Sphere::SetRadious(const float& radious)
	{
		m_radious = std::max(radious, std::numeric_limits<float>::epsilon());
		RequestUpdate();
	}
	void Sphere::SetColor(const Graphics::Color& color)
	{
		m_color = color;
		RequestUpdate();
	}

	float Sphere::GetRadious() const noexcept
	{
		return m_radious;
	}
	Graphics::Color& Sphere::GetColor() noexcept
	{
		return m_color;
	}
	const Graphics::Color& Sphere::GetColor() const noexcept
	{
		return m_color;
	}
	const Texture* Sphere::GetTexture() const
	{
		return m_pTexture;
	}
}