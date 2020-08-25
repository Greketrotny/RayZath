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


	// [STRUCT] Triangle --------------------------|
	// -- constructor -- //
	Triangle::Triangle(Vertex* v1, Vertex* v2, Vertex* v3,
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

	// -- methods -- //
	const Graphics::Color& Triangle::Color()
	{
		return color;
	}
	void Triangle::Color(const Graphics::Color& newColor)
	{
		this->color = newColor;
	}
	const Math::vec3<float>& Triangle::GetNormal() const
	{
		return normal;
	}

	// vertices
	Triangle::Vertex& Triangle::V1() const
	{
		return *v1;
	}
	void Triangle::V1(const Vertex& newVertex)
	{
		*this->v1 = newVertex;
	}
	void Triangle::V1(const float& x, const float& y, const float& z)
	{
		this->v1->x = x;
		this->v1->y = y;
		this->v1->z = z;
	}
	Triangle::Vertex& Triangle::V2() const
	{
		return *v2;
	}
	void Triangle::V2(const Vertex& newVertex)
	{
		*this->v2 = newVertex;
	}
	void Triangle::V2(const float& x, const float& y, const float& z)
	{
		this->v2->x = x;
		this->v2->y = y;
		this->v2->z = z;
	}
	Triangle::Vertex& Triangle::V3() const
	{
		return *v3;
	}
	void Triangle::V3(const Vertex& newVertex)
	{
		*this->v3 = newVertex;
	}
	void Triangle::V3(const float& x, const float& y, const float& z)
	{
		this->v3->x = x;
		this->v3->y = y;
		this->v3->z = z;
	}

	// texture coordinates
	Texcrd& Triangle::T1() const
	{
		return *t1;
	}
	void Triangle::T1(const Texcrd& newTexcds)
	{
		*this->t1 = newTexcds;
	}
	void Triangle::T1(const float& u, const float& v)
	{
		this->t1->u = u;
		this->t1->v = v;
	}
	Texcrd& Triangle::T2() const
	{
		return *t2;
	}
	void Triangle::T2(const Texcrd& newTexcds)
	{
		*this->t2 = newTexcds;
	}
	void Triangle::T2(const float& u, const float& v)
	{
		this->t2->u = v;
		this->t2->v = u;
	}
	Texcrd& Triangle::T3() const
	{
		return *t3;
	}
	void Triangle::T3(const Texcrd& newTexcds)
	{
		*this->t3 = newTexcds;
	}
	void Triangle::T3(const float& u, const float& v)
	{
		this->t3->u = u;
		this->t3->v = v;
	}
	// --------------------------------------------|
}