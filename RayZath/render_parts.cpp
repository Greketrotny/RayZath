#include "render_parts.h"

#include <algorithm>

namespace RayZath
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
		const Math::vec3f& center,
		const Math::vec3f& scale)
		: m_position(position)
		, m_rotation(rotation)
		, m_center(center)
		, m_scale(scale)
		, m_coord_system(rotation)
	{}

	void Transformation::LookAtPoint(const Math::vec3f& point, const Math::angle_radf& angle)
	{
		LookInDirection(point - m_position);
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
	const Math::vec3f& Transformation::GetCenter() const
	{
		return m_center;
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
	void Transformation::SetCenter(const Math::vec3f& center)
	{
		m_center = center;
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



	// ~~~~~~~~ [STRUCT] Texture ~~~~~~~~
	Texture::Texture(
		Updatable* updatable,
		const ConStruct<Texture>& con_struct)
		: WorldObject(updatable, con_struct)
		, m_bitmap(con_struct.bitmap)
		, m_filter_mode(con_struct.filter_mode)
		, m_address_mode(con_struct.address_mode)
	{}

	const Graphics::Bitmap& Texture::GetBitmap() const noexcept
	{
		return m_bitmap;
	}
	Texture::FilterMode Texture::GetFilterMode() const noexcept
	{
		return m_filter_mode;
	}
	Texture::AddressMode Texture::GetAddressMode() const noexcept
	{
		return m_address_mode;
	}

	void Texture::SetBitmap(const Graphics::Bitmap& bitmap)
	{
		m_bitmap = bitmap;
		GetStateRegister().RequestUpdate();
	}
	void Texture::SetFilterMode(const FilterMode filter_mode)
	{
		m_filter_mode = filter_mode;
		GetStateRegister().RequestUpdate();
	}
	void Texture::SetAddressMode(const AddressMode address_mode)
	{
		m_address_mode = address_mode;
		GetStateRegister().RequestUpdate();
	}
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



	// ~~~~~~~~ [STRUCT] Triangle ~~~~~~~~
	Triangle::Triangle(
		Vertex* v1, Vertex* v2, Vertex* v3,
		Texcrd* t1, Texcrd* t2, Texcrd* t3,
		Normal* n1, Normal* n2, Normal* n3,
		const uint32_t& mat_id)
	{
		this->v1 = v1;
		this->v2 = v2;
		this->v3 = v3;

		this->t1 = t1;
		this->t2 = t2;
		this->t3 = t3;

		this->n1 = n1;
		this->n2 = n2;
		this->n3 = n3;

		this->material_id = mat_id;
	}
	Triangle::~Triangle()
	{
		v1 = nullptr;
		v2 = nullptr;
		v3 = nullptr;

		t1 = nullptr;
		t2 = nullptr;
		t3 = nullptr;
	}

	void Triangle::CalculateNormal()
	{
		normal = Math::vec3f::CrossProduct(*v2 - *v3, *v2 - *v1);
		normal.Normalize();
	}
	BoundingBox Triangle::GetBoundingBox() const
	{
		return BoundingBox(*v1, *v2, *v3);
	}
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
}