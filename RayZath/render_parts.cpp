#include "render_parts.hpp"

#include <algorithm>

namespace RayZath::Engine
{
	// ~~~~~~~~ [STRUCT] CoordSystem ~~~~~~~~
	CoordSystem::CoordSystem()
	{
		x_axis = Math::vec3f(1.0f, 0.0f, 0.0f);
		y_axis = Math::vec3f(0.0f, 1.0f, 0.0f);
		z_axis = Math::vec3f(0.0f, 0.0f, 1.0f);
	}
	CoordSystem::CoordSystem(const Math::vec3f& rotation)
	{
		ApplyRotation(rotation);
	}

	CoordSystem& CoordSystem::operator*=(const CoordSystem& other)
	{
		x_axis = other.TransformForward(x_axis);
		y_axis = other.TransformForward(y_axis);
		z_axis = other.TransformForward(z_axis);
		return *this;
	}

	const Math::vec3f CoordSystem::GetXAxis() const
	{
		return x_axis;
	}
	const Math::vec3f CoordSystem::GetYAxis() const
	{
		return y_axis;
	}
	const Math::vec3f CoordSystem::GetZAxis() const
	{
		return z_axis;
	}

	Math::vec3f CoordSystem::TransformForward(const Math::vec3f& v) const
	{
		return x_axis * v.x + y_axis * v.y + z_axis * v.z;
	}
	Math::vec3f CoordSystem::TransformBackward(const Math::vec3f& v) const
	{
		return Math::vec3f(
			x_axis.x * v.x + x_axis.y * v.y + x_axis.z * v.z,
			y_axis.x * v.x + y_axis.y * v.y + y_axis.z * v.z,
			z_axis.x * v.x + z_axis.y * v.y + z_axis.z * v.z);
	}
	void CoordSystem::ApplyRotation(const Math::vec3f& rotation)
	{
		x_axis = Math::vec3f(1.0f, 0.0f, 0.0f).RotatedXYZ(rotation);
		y_axis = Math::vec3f(0.0f, 1.0f, 0.0f).RotatedXYZ(rotation);
		z_axis = Math::vec3f(0.0f, 0.0f, 1.0f).RotatedXYZ(rotation);
	}
	void CoordSystem::LookAt(const Math::vec3f& rotation)
	{
		x_axis = Math::vec3f(1.0f, 0.0f, 0.0f).RotatedZ(rotation.z).RotatedX(rotation.x).RotatedY(rotation.y);
		y_axis = Math::vec3f(0.0f, 1.0f, 0.0f).RotatedZ(rotation.z).RotatedX(rotation.x).RotatedY(rotation.y);
		z_axis = Math::vec3f(0.0f, 0.0f, 1.0f).RotatedZ(rotation.z).RotatedX(rotation.x).RotatedY(rotation.y);
	}
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



	// ~~~~~~~~ [STRUCT] Transformation ~~~~~~~~
	Transformation::Transformation(
		const Math::vec3f& position,
		const Math::vec3f& rotation,
		const Math::vec3f& scale)
		: m_position(position)
		, m_rotation(rotation)
		, m_scale(scale)
		, m_coord_system(rotation)
	{}

	Transformation& Transformation::operator*=(const Transformation& other)
	{
		m_position = other.GetCoordSystem().TransformForward(m_position);
		m_position += other.GetPosition();
		m_coord_system *= other.GetCoordSystem();
		m_scale *= other.m_scale;
		return *this;
	}

	void Transformation::LookAtPoint(const Math::vec3f& point, const Math::angle_radf& angle)
	{
		LookInDirection(point - m_position);
		m_rotation.z = angle.value();
	}
	void Transformation::LookInDirection(const Math::vec3f& direction, const Math::angle_radf& angle)
	{
		const Math::vec3f dir = direction.Normalized();
		const float x_angle = asin(dir.y);
		const float y_angle = -atan2f(dir.x, dir.z);
		m_rotation = Math::vec3f(x_angle, y_angle, angle.value());
		m_coord_system.LookAt(m_rotation);
	}
	
	const Math::vec3f& Transformation::GetPosition() const
	{
		return m_position;
	}
	const Math::vec3f& Transformation::GetRotation() const
	{
		return m_rotation;
	}
	const Math::vec3f& Transformation::GetScale() const
	{
		return m_scale;
	}
	const CoordSystem& Transformation::GetCoordSystem() const
	{
		return m_coord_system;
	}

	void Transformation::SetPosition(const Math::vec3f& position)
	{
		m_position = position;
	}
	void Transformation::SetRotation(const Math::vec3f& rotation)
	{
		m_rotation = rotation;
		m_coord_system.ApplyRotation(rotation);
	}
	void Transformation::SetScale(const Math::vec3f& scale)
	{
		m_scale = scale;
	}
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


	// ~~~~~~~~ [STRUCT] BoundingBox ~~~~~~~~~
	BoundingBox::BoundingBox(
		const Math::vec3f& p1,
		const Math::vec3f& p2)
	{
		min.x = std::min(p1.x, p2.x);
		min.y = std::min(p1.y, p2.y);
		min.z = std::min(p1.z, p2.z);
					   		 
		max.x = std::max(p1.x, p2.x);
		max.y = std::max(p1.y, p2.y);
		max.z = std::max(p1.z, p2.z);
	}
	BoundingBox::BoundingBox(
		const Math::vec3f& p1,
		const Math::vec3f& p2,
		const Math::vec3f& p3)
		: BoundingBox(p1, p2)
	{
		ExtendBy(p3);
	}

	void BoundingBox::Reset(const Math::vec3f& point)
	{
		min = point;
		max = point;
	}
	void BoundingBox::ExtendBy(const Math::vec3f& point)
	{
		if (min.x > point.x) min.x = point.x;
		if (min.y > point.y) min.y = point.y;
		if (min.z > point.z) min.z = point.z;
		if (max.x < point.x) max.x = point.x;
		if (max.y < point.y) max.y = point.y;
		if (max.z < point.z) max.z = point.z;
	}
	void BoundingBox::ExtendBy(const BoundingBox& bb)
	{
		if (min.x > bb.min.x) min.x = bb.min.x;
		if (min.y > bb.min.y) min.y = bb.min.y;
		if (min.z > bb.min.z) min.z = bb.min.z;
		if (max.x < bb.max.x) max.x = bb.max.x;
		if (max.y < bb.max.y) max.y = bb.max.y;
		if (max.z < bb.max.z) max.z = bb.max.z;
	}
	Math::vec3f BoundingBox::GetCentroid() const noexcept
	{
		return (min + max) * 0.5f;
	}
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
}