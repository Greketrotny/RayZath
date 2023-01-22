#include "render_parts.hpp"
#include "cpu_render_utils.hpp"

#include <algorithm>

namespace RayZath::Engine
{
	CoordSystem::CoordSystem()
	{
		x_axis = Math::vec3f(1.0f, 0.0f, 0.0f);
		y_axis = Math::vec3f(0.0f, 1.0f, 0.0f);
		z_axis = Math::vec3f(0.0f, 0.0f, 1.0f);
	}
	CoordSystem::CoordSystem(const Math::vec3f& rotation)
	{
		applyRotation(rotation);
	}

	CoordSystem& CoordSystem::operator*=(const CoordSystem& other)
	{
		x_axis = other.transformForward(x_axis);
		y_axis = other.transformForward(y_axis);
		z_axis = other.transformForward(z_axis);
		return *this;
	}

	const Math::vec3f CoordSystem::xAxis() const
	{
		return x_axis;
	}
	const Math::vec3f CoordSystem::yAxis() const
	{
		return y_axis;
	}
	const Math::vec3f CoordSystem::zAxis() const
	{
		return z_axis;
	}

	[[nodiscard]] Math::vec3f CoordSystem::transformForward(const Math::vec3f& v) const
	{
		return x_axis * v.x + y_axis * v.y + z_axis * v.z;
	}
	[[nodiscard]] Math::vec3f CoordSystem::transformBackward(const Math::vec3f& v) const
	{
		return Math::vec3f(
			x_axis.x * v.x + x_axis.y * v.y + x_axis.z * v.z,
			y_axis.x * v.x + y_axis.y * v.y + y_axis.z * v.z,
			z_axis.x * v.x + z_axis.y * v.y + z_axis.z * v.z);
	}
	void CoordSystem::applyRotation(const Math::vec3f& rotation)
	{
		x_axis = Math::vec3f(1.0f, 0.0f, 0.0f).RotatedXYZ(rotation);
		y_axis = Math::vec3f(0.0f, 1.0f, 0.0f).RotatedXYZ(rotation);
		z_axis = Math::vec3f(0.0f, 0.0f, 1.0f).RotatedXYZ(rotation);
	}
	void CoordSystem::lookAt(const Math::vec3f& rotation)
	{
		x_axis = Math::vec3f(1.0f, 0.0f, 0.0f).RotatedZ(rotation.z).RotatedX(rotation.x).RotatedY(rotation.y);
		y_axis = Math::vec3f(0.0f, 1.0f, 0.0f).RotatedZ(rotation.z).RotatedX(rotation.x).RotatedY(rotation.y);
		z_axis = Math::vec3f(0.0f, 0.0f, 1.0f).RotatedZ(rotation.z).RotatedX(rotation.x).RotatedY(rotation.y);
	}


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
		m_position = other.coordSystem().transformForward(m_position);
		m_position += other.position();
		m_coord_system *= other.coordSystem();
		m_scale *= other.m_scale;
		return *this;
	}

	void Transformation::lookAtPoint(const Math::vec3f& point, const Math::angle_radf& angle)
	{
		lookInDirection(point - m_position);
		m_rotation.z = angle.value();
	}
	void Transformation::lookInDirection(const Math::vec3f& direction, const Math::angle_radf& angle)
	{
		const Math::vec3f dir = direction.Normalized();
		const float x_angle = asin(dir.y);
		const float y_angle = -atan2f(dir.x, dir.z);
		m_rotation = Math::vec3f(x_angle, y_angle, angle.value());
		m_coord_system.lookAt(m_rotation);
	}

	void Transformation::transformG2L(CPU::RangedRay& ray) const
	{
		ray.origin -= position();
		ray.origin = m_coord_system.transformBackward(ray.origin);
		ray.origin /= scale();

		ray.direction = m_coord_system.transformBackward(ray.direction);
		ray.direction /= scale();
	}
	void Transformation::transformL2G(Math::vec3f32& v) const
	{
		v /= scale();
		v = m_coord_system.transformForward(v);
	}
	void Transformation::transformL2GNoScale(Math::vec3f32& v) const
	{
		v = m_coord_system.transformForward(v);
	}

	const Math::vec3f& Transformation::position() const
	{
		return m_position;
	}
	const Math::vec3f& Transformation::rotation() const
	{
		return m_rotation;
	}
	const Math::vec3f& Transformation::scale() const
	{
		return m_scale;
	}
	const CoordSystem& Transformation::coordSystem() const
	{
		return m_coord_system;
	}

	void Transformation::position(const Math::vec3f& position)
	{
		m_position = position;
	}
	void Transformation::rotation(const Math::vec3f& rotation)
	{
		m_rotation = rotation;
		m_coord_system.applyRotation(rotation);
	}
	void Transformation::scale(const Math::vec3f& scale)
	{
		m_scale = scale;
	}


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
		extendBy(p3);
	}

	void BoundingBox::reset(const Math::vec3f& point)
	{
		min = point;
		max = point;
	}
	void BoundingBox::extendBy(const Math::vec3f& point)
	{
		if (min.x > point.x) min.x = point.x;
		if (min.y > point.y) min.y = point.y;
		if (min.z > point.z) min.z = point.z;
		if (max.x < point.x) max.x = point.x;
		if (max.y < point.y) max.y = point.y;
		if (max.z < point.z) max.z = point.z;
	}
	void BoundingBox::extendBy(const BoundingBox& bb)
	{
		if (min.x > bb.min.x) min.x = bb.min.x;
		if (min.y > bb.min.y) min.y = bb.min.y;
		if (min.z > bb.min.z) min.z = bb.min.z;
		if (max.x < bb.max.x) max.x = bb.max.x;
		if (max.y < bb.max.y) max.y = bb.max.y;
		if (max.z < bb.max.z) max.z = bb.max.z;
	}
	Math::vec3f BoundingBox::centroid() const noexcept
	{
		return (min + max) * 0.5f;
	}
	bool BoundingBox::rayIntersection(const CPU::RangedRay& ray) const
	{
 		float t1 = (min.x - ray.origin.x) / ray.direction.x;
		float t2 = (max.x - ray.origin.x) / ray.direction.x;
		float t3 = (min.y - ray.origin.y) / ray.direction.y;
		float t4 = (max.y - ray.origin.y) / ray.direction.y;
		float t5 = (min.z - ray.origin.z) / ray.direction.z;
		float t6 = (max.z - ray.origin.z) / ray.direction.z;

		auto my_min = [](const float a, const float b) {
			return a < b ? a : b;
		};
		auto my_max = [](const float a, const float b) {
			return a > b ? a : b;
		};

		float tmin = my_max(my_max(my_min(t1, t2), my_min(t3, t4)), my_min(t5, t6));
		float tmax = my_min(my_min(my_max(t1, t2), my_max(t3, t4)), my_max(t5, t6));

		return !(tmax < ray.near_far.x || tmin > tmax || tmin > ray.near_far.y);
	}
}