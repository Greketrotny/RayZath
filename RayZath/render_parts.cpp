#include "render_parts.h"
#include <algorithm>

namespace RayZath
{
	// ~~~~~~~~ [STRUCT] Material ~~~~~~~~
	Material::Material(const MaterialType& type, const float& emitance)
		: m_material_type(type)
		, m_emitance(emitance)
	{
		if (m_emitance < 0.0f) m_emitance = 0.0f;
	}
	Material::Material(const Material& material)
		: m_material_type(material.m_material_type)
		, m_emitance(material.m_emitance)
	{}
	Material::~Material()
	{}

	Material& Material::operator=(const Material& material)
	{
		m_material_type = material.m_material_type;
		m_emitance = material.m_emitance;
		return *this;
	}

	void Material::Set(const MaterialType& type, const float& emitance)
	{
		SetMaterialType(type);
		SetEmitance(emitance);
	}
	void Material::SetMaterialType(const MaterialType& type)
	{
		m_material_type = type;
	}
	void Material::SetEmitance(const float& emitance)
	{
		m_emitance = std::max(emitance, 0.0f);
	}

	MaterialType Material::GetMaterialType() const noexcept
	{
		return m_material_type;
	}
	float Material::GetEmitance() const noexcept
	{
		return m_emitance;
	}
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
}