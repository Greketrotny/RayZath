#include "Sphere.h"

namespace RayZath::Engine
{
	Sphere::Sphere(
		Updatable* updatable,
		const ConStruct<Sphere>& conStruct)
		: RenderObject(updatable, conStruct)
		, m_material(std::bind(&Sphere::NotifyMaterial, this))
	{
		SetRadius(conStruct.radius);
		SetMaterial(conStruct.material);
	}
	Sphere::~Sphere()
	{
	}


	void Sphere::SetRadius(const float& radius)
	{
		m_radius = std::max(radius, std::numeric_limits<float>::epsilon());
		GetStateRegister().RequestUpdate();
	}
	float Sphere::GetRadius() const noexcept
	{
		return m_radius;
	}

	void Sphere::SetMaterial(const Handle<Material>& material)
	{
		m_material = material;
		GetStateRegister().RequestUpdate();
	}
	const Handle<Material>& Sphere::GetMaterial() const
	{
		return static_cast<const Handle<Material>&>(m_material);
	}

	void Sphere::Update()
	{
		// [>] Update AABB
		auto bit = [](const unsigned int& n, const unsigned int& b) -> bool
		{
			return (n >> b) & 0x1u;
		};

		// construct 8 AABB vertices and transpose them to sphere space
		Math::vec3f P[8];
		for (unsigned int i = 0; i < 8; i++)
		{
			P[i] =
				(Math::vec3f(bit(i, 2), bit(i, 1), bit(i, 0)) -
					Math::vec3f(0.5f, 0.5f, 0.5f)) *
				m_radius * 2.0f;

			P[i] += GetTransformation().GetCenter();
			P[i] *= GetTransformation().GetScale();
			P[i].RotateXYZ(GetTransformation().GetRotation());
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
		m_bounding_box.min += GetTransformation().GetPosition();
		m_bounding_box.max += GetTransformation().GetPosition();
	}
	void Sphere::NotifyMaterial()
	{
		GetStateRegister().RequestUpdate();
	}
}