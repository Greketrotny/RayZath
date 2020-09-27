#include "render_parts.h"
#include <algorithm>

namespace RayZath
{
	// ~~~~~~~~ [STRUCT] Material ~~~~~~~~
	Material::Material(
		const float& reflectance,
		const float& glossiness,
		const float& transmitance,
		const float& ior,
		const float& emitance)
	{
		SetReflectance(reflectance);
		SetGlossiness(glossiness);
		SetTransmitance(transmitance);
		SetIndexOfRefraction(ior);
		SetEmitance(emitance);
	}
	Material::Material(const Material& material)
		: m_reflectance(material.m_reflectance)
		, m_glossiness(material.m_glossiness)
		, m_transmitance(material.m_transmitance)
		, m_ior(material.m_ior)
		, m_emitance(material.m_emitance)
	{}
	Material::~Material()
	{}

	Material& Material::operator=(const Material& material)
	{
		m_reflectance = material.m_reflectance;
		m_glossiness = material.m_glossiness;
		m_transmitance = material.m_transmitance;
		m_ior = material.m_ior;
		m_emitance = material.m_emitance;
		return *this;
	}

	void Material::SetReflectance(const float& reflectance)
	{
		m_reflectance = std::clamp(reflectance, 0.0f, 1.0f);
	}
	void Material::SetGlossiness(const float& glossiness)
	{
		m_glossiness = std::clamp(glossiness, 0.0f, 1.0f);
	}
	void Material::SetTransmitance(const float& transmitance)
	{
		m_transmitance = std::clamp(transmitance, 0.0f, 1.0f);
	}
	void Material::SetIndexOfRefraction(const float& ior)
	{
		m_ior = std::max(ior, 1.0f);
	}
	void Material::SetEmitance(const float& emitance)
	{
		m_emitance = std::max(emitance, 0.0f);
	}

	
	float Material::GetReflectance() const noexcept
	{
		return m_reflectance;
	}
	float Material::GetGlossiness() const noexcept
	{
		return m_glossiness;
	}
	float Material::GetTransmitance() const noexcept
	{
		return m_transmitance;
	}
	float Material::GetIndexOfRefraction() const noexcept
	{
		return m_ior;
	}
	float Material::GetEmitance() const noexcept
	{
		return m_emitance;
	}
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



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
	Texture::Texture(const Texture& texture)
		: m_bitmap(texture.m_bitmap)
		, m_filter_mode(texture.m_filter_mode)
	{}
	Texture::Texture(Texture&& texture) noexcept
		: m_bitmap(std::move(texture.m_bitmap))
		, m_filter_mode(texture.m_filter_mode)
	{}
	Texture::Texture(size_t width, size_t height, Texture::FilterMode filter_mode)
		: m_bitmap(width, height)
		, m_filter_mode(filter_mode)
	{}
	Texture::Texture(const Graphics::Bitmap& bitmap, Texture::FilterMode filter_mode)
		: m_bitmap(bitmap)
		, m_filter_mode(filter_mode)
	{}
	Texture::~Texture()
	{}

	Texture& Texture::operator=(const Texture& texture)
	{
		m_bitmap = texture.m_bitmap;
		m_filter_mode = texture.m_filter_mode;

		return *this;
	}
	Texture& Texture::operator=(Texture&& texture) noexcept
	{
		if (&texture == this)
			return *this;

		m_bitmap = std::move(texture.m_bitmap);
		m_filter_mode = texture.m_filter_mode;

		return *this;
	}

	const Graphics::Bitmap& Texture::GetBitmap() const noexcept
	{
		return m_bitmap;
	}
	Texture::FilterMode Texture::GetFilterMode() const noexcept
	{
		return m_filter_mode;
	}
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



	// ~~~~~~~~ [STRUCT] Triangle ~~~~~~~~
	Triangle::Triangle(
		Vertex* v1, Vertex* v2, Vertex* v3,
		Texcrd* t1, Texcrd* t2, Texcrd* t3,
		Graphics::Color color)
	{
		this->v1 = v1;
		this->v2 = v2;
		this->v3 = v3;

		this->t1 = t1;
		this->t2 = t2;
		this->t3 = t3;

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
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
}