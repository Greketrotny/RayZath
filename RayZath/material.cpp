#include "material.h"

#include <algorithm>

namespace RayZath
{
	// ~~~~~~~~ [STRUCT] Material ~~~~~~~~
	Material::Material(
		Updatable* updatable, 
		const ConStruct<Material>& con_struct)
		: WorldObject(updatable, ConStruct<WorldObject>())
		, m_texture(con_struct.texture, std::bind(&Material::ResourceNotify, this))
	{
		SetColor(con_struct.color);
		SetReflectance(con_struct.reflectance);
		SetGlossiness(con_struct.glossiness);
		SetTransmittance(con_struct.transmittance);
		SetIndexOfRefraction(con_struct.ior);
		SetEmittance(con_struct.emittance);
		SetScattering(con_struct.scattering);
	}
	Material::~Material()
	{}

	void Material::SetColor(const Graphics::Color& color)
	{
		m_color = color;
		GetStateRegister().MakeModified();
	}
	void Material::SetReflectance(const float& reflectance)
	{
		m_reflectance = std::clamp(reflectance, 0.0f, 1.0f);
		GetStateRegister().MakeModified();
	}
	void Material::SetGlossiness(const float& glossiness)
	{
		m_glossiness = std::clamp(glossiness, 0.0f, 1.0f);
		GetStateRegister().MakeModified();
	}
	void Material::SetTransmittance(const float& transmittance)
	{
		m_transmittance = std::clamp(transmittance, 0.0f, 1.0f);
		GetStateRegister().MakeModified();
	}
	void Material::SetIndexOfRefraction(const float& ior)
	{
		m_ior = std::max(ior, 1.0f);
		GetStateRegister().MakeModified();
	}
	void Material::SetEmittance(const float& emittance)
	{
		m_emittance = std::max(emittance, 0.0f);
		GetStateRegister().MakeModified();
	}
	void Material::SetScattering(const float& scattering)
	{
		m_scattering = std::max(0.0f, scattering);
		GetStateRegister().MakeModified();
	}

	void Material::SetTexture(const Handle<Texture>& texture)
	{
		m_texture = texture;
		GetStateRegister().MakeModified();
	}
	void Material::SetEmittanceMap(const Handle<EmittanceMap>& emittance_map)
	{
		m_emittance_map = emittance_map;
		GetStateRegister().MakeModified();
	}

	const Graphics::Color& Material::GetColor() const noexcept
	{
		return m_color;
	}
	float Material::GetReflectance() const noexcept
	{
		return m_reflectance;
	}
	float Material::GetGlossiness() const noexcept
	{
		return m_glossiness;
	}
	float Material::GetTransmittance() const noexcept
	{
		return m_transmittance;
	}
	float Material::GetIndexOfRefraction() const noexcept
	{
		return m_ior;
	}
	float Material::GetEmittance() const noexcept
	{
		return m_emittance;
	}
	float Material::GetScattering() const noexcept
	{
		return m_scattering;
	}

	const Handle<Texture>& Material::GetTexture() const
	{
		return static_cast<const Handle<Texture>&>(m_texture);
	}
	const Handle<EmittanceMap>& Material::GetEmittanceMap() const
	{
		return m_emittance_map;
	}

	void Material::ResourceNotify()
	{
		GetStateRegister().MakeModified();
	}
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
}