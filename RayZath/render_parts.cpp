#include "render_parts.h"
#include <algorithm>

namespace RayZath
{
	// ~~~~~~~~ [STRUCT] Material ~~~~~~~~
	Material::Material(
		const MaterialType& type, 
		const float& emitance,
		const float& reflectance)
		: m_material_type(type)
	{
		SetEmitance(emitance);
		SetReflectance(reflectance);
	}
	Material::Material(const Material& material)
		: m_material_type(material.m_material_type)
		, m_emitance(material.m_emitance)
		, m_reflectance(material.m_reflectance)
	{}
	Material::~Material()
	{}

	Material& Material::operator=(const Material& material)
	{
		m_material_type = material.m_material_type;
		m_emitance = material.m_emitance;
		m_reflectance = material.m_reflectance;
		return *this;
	}

	void Material::Set(
		const MaterialType& type,
		const float& emitance,
		const float& reflectance)
	{
		SetMaterialType(type);
		SetEmitance(emitance);
		SetReflectance(reflectance);
	}
	void Material::SetMaterialType(const MaterialType& type)
	{
		m_material_type = type;
	}
	void Material::SetEmitance(const float& emitance)
	{
		m_emitance = std::max(emitance, 0.0f);
	}
	void Material::SetReflectance(const float& reflectance)
	{
		m_reflectance = std::clamp(reflectance, 0.0f, 1.0f);
	}

	MaterialType Material::GetMaterialType() const noexcept
	{
		return m_material_type;
	}
	float Material::GetEmitance() const noexcept
	{
		return m_emitance;
	}
	float Material::GetReflectance() const noexcept
	{
		return m_reflectance;
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
}