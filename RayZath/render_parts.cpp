#include "render_parts.h"

#include <algorithm>

namespace RayZath
{
	// ~~~~~~~~ [STRUCT] BoundingBox ~~~~~~~~~
	BoundingBox::BoundingBox(
		const Math::vec3<float>& p1,
		const Math::vec3<float>& p2)
	{
		min.x = std::min(p1.x, p2.x);
		min.y = std::min(p1.y, p2.y);
		min.z = std::min(p1.z, p2.z);
					   		 
		max.x = std::max(p1.x, p2.x);
		max.y = std::max(p1.y, p2.y);
		max.z = std::max(p1.z, p2.z);
	}
	BoundingBox::BoundingBox(
		const Math::vec3<float>& p1,
		const Math::vec3<float>& p2,
		const Math::vec3<float>& p3)
		: BoundingBox(p1, p2)
	{
		ExtendBy(p3);
	}

	void BoundingBox::Reset(const Math::vec3<float>& point)
	{
		min = point;
		max = point;
	}
	void BoundingBox::ExtendBy(const Math::vec3<float>& point)
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
	Math::vec3<float> BoundingBox::GetCentroid() const noexcept
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
	{}

	const Graphics::Bitmap& Texture::GetBitmap() const noexcept
	{
		return m_bitmap;
	}
	Texture::FilterMode Texture::GetFilterMode() const noexcept
	{
		return m_filter_mode;
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
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



	// ~~~~~~~~ [STRUCT] Triangle ~~~~~~~~
	Triangle::Triangle(
		Vertex* v1, Vertex* v2, Vertex* v3,
		Texcrd* t1, Texcrd* t2, Texcrd* t3,
		Normal* n1, Normal* n2, Normal* n3,
		Graphics::Color color)
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

		this->color = color;
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
		normal = Math::vec3<float>::CrossProduct(*v2 - *v3, *v2 - *v1);
		normal.Normalize();
	}
	BoundingBox Triangle::GetBoundingBox() const
	{
		return BoundingBox(*v1, *v2, *v3);
	}
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
}