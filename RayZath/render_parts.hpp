#ifndef RENDER_PARTS_H
#define RENDER_PARTS_H

#include "vec3.h"
#include "angle.h"

#include <cmath>
#include <algorithm>

namespace RayZath::Engine::CPU
{
	struct RangedRay;
}

namespace RayZath::Engine
{
	class CoordSystem
	{
	private:
		Math::vec3f x_axis, y_axis, z_axis;

	public:
		CoordSystem();
		CoordSystem(const Math::vec3f& rotation);

		CoordSystem& operator*=(const CoordSystem& other);

		const Math::vec3f xAxis() const;
		const Math::vec3f yAxis() const;
		const Math::vec3f zAxis() const;

		Math::vec3f transformForward(const Math::vec3f& v) const;
		Math::vec3f transformBackward(const Math::vec3f& v) const;
		void applyRotation(const Math::vec3f& rotation);
		void lookAt(const Math::vec3f& rotation);
	};
	class Transformation
	{
	private:
		Math::vec3f m_position, m_rotation, m_scale;
		CoordSystem m_coord_system;

	public:
		Transformation(
			const Math::vec3f& position = Math::vec3f(0.0f),
			const Math::vec3f& rotation = Math::vec3f(0.0f),
			const Math::vec3f& scale = Math::vec3f(1.0f));

		Transformation& operator*=(const Transformation& other);

		void lookAtPoint(const Math::vec3f& point, const Math::angle_radf& angle = 0.0f);
		void lookInDirection(const Math::vec3f& direction, const Math::angle_radf& angle = 0.0f);

		void transformG2L(CPU::RangedRay& ray) const;
		void transformL2G(Math::vec3f32& v) const;
		void transformL2GNoScale(Math::vec3f32& v) const;

		const Math::vec3f& position() const;
		const Math::vec3f& rotation() const;
		const Math::vec3f& scale() const;
		const CoordSystem& coordSystem() const;

		void position(const Math::vec3f& position);
		void rotation(const Math::vec3f& rotation);
		void scale(const Math::vec3f& scale);
	};
	class BoundingBox
	{
	public:
		Math::vec3f min, max;


	public:
		BoundingBox(
			const Math::vec3f& p1 = Math::vec3f(0.0f, 0.0f, 0.0f),
			const Math::vec3f& p2 = Math::vec3f(0.0f, 0.0f, 0.0f));
		BoundingBox(
			const Math::vec3f& p1,
			const Math::vec3f& p2,
			const Math::vec3f& p3);


		void reset(const Math::vec3f& point = Math::vec3f(0.0f, 0.0f, 0.0f));
		void extendBy(const Math::vec3f& point);
		void extendBy(const BoundingBox& bb);

		Math::vec3f centroid() const noexcept;

		bool rayIntersection(const CPU::RangedRay& ray) const;
	};
}

#endif // !RENDER_PARTS_H
