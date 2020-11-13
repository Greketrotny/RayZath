#include "Sphere.h"

namespace RayZath
{
	Sphere::Sphere(
		const size_t& id,
		Updatable* updatable,
		const ConStruct<Sphere>& conStruct)
		: RenderObject(id, updatable, conStruct)
	{
		SetRadius(conStruct.radius);
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
		GetStateRegister().MakeModified();
	}
	void Sphere::UnloadTexture()
	{
		if (m_pTexture) delete m_pTexture;
		m_pTexture = nullptr;
		GetStateRegister().MakeModified();
	}
	void Sphere::SetRadius(const float& radius)
	{
		m_radius = std::max(radius, std::numeric_limits<float>::epsilon());
		GetStateRegister().RequestUpdate();
	}
	void Sphere::SetColor(const Graphics::Color& color)
	{
		m_color = color;
		GetStateRegister().RequestUpdate();
	}

	float Sphere::GetRadius() const noexcept
	{
		return m_radius;
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
				m_radius * 2.0f;

			P[i] += m_center;
			P[i] *= m_scale;
			P[i].RotateXYZ(m_rotation);
		}

		// expand bounding volume extents
		m_bounding_box.Reset();
		for (unsigned int i = 0; i < 8; i++)
		{
			m_bounding_box.min.x = std::min(m_bounding_box.min.x, P[i].x);
			m_bounding_box.min.y = std::min(m_bounding_box.min.y, P[i].y);
			m_bounding_box.min.z = std::min(m_bounding_box.min.z, P[i].z);
			m_bounding_box.max.x = std::max(m_bounding_box.max.x, P[i].x);
			m_bounding_box.max.y = std::max(m_bounding_box.max.y, P[i].y);
			m_bounding_box.max.z = std::max(m_bounding_box.max.z, P[i].z);
		}

		// transpose BB by sphere position
		m_bounding_box.min += m_position;
		m_bounding_box.max += m_position;
	}
}