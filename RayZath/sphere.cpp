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

	void Sphere::Update()
	{
		// [>] Update AABB
		auto bit = [](const unsigned int& n, const unsigned int& b) -> bool
		{
			return (n >> b) & 0x1u;
		};

		// construct 8 AABB vertices and transpose them to sphere space
		Math::vec3<float> P[8];
		for (unsigned int i = 0; i < 8; i++)
		{
			P[i] =
				(Math::vec3<float>(bit(i, 2), bit(i, 1), bit(i, 0)) -
					Math::vec3<float>(0.5f, 0.5f, 0.5f)) *
				m_radious * 2.0f;

			P[i] += m_center;
			P[i] *= m_scale;
			P[i].RotateXYZ(m_rotation);
		}

		// expand bounding volume extents
		m_bounding_volume.Reset();
		for (unsigned int i = 0; i < 8; i++)
		{
			m_bounding_volume.min.x = std::min(m_bounding_volume.min.x, P[i].x);
			m_bounding_volume.min.y = std::min(m_bounding_volume.min.y, P[i].y);
			m_bounding_volume.min.z = std::min(m_bounding_volume.min.z, P[i].z);
			m_bounding_volume.max.x = std::max(m_bounding_volume.max.x, P[i].x);
			m_bounding_volume.max.y = std::max(m_bounding_volume.max.y, P[i].y);
			m_bounding_volume.max.z = std::max(m_bounding_volume.max.z, P[i].z);
		}

		// transpose BB by sphere position
		m_bounding_volume.min += m_position;
		m_bounding_volume.max += m_position;
	}
}