#include "material.h"

#include <algorithm>

namespace RayZath
{
	// ~~~~~~~~ [STRUCT] Material ~~~~~~~~
	Material::Material(
		Updatable* updatable, 
		const ConStruct<Material>& con_struct)
		: WorldObject(updatable, con_struct)
		, m_texture(con_struct.texture, std::bind(&Material::ResourceNotify, this))
		, m_normal_map(con_struct.normal_map, std::bind(&Material::ResourceNotify, this))
		, m_metalic_map(con_struct.metalic_map, std::bind(&Material::ResourceNotify, this))
		, m_specular_map(con_struct.specular_map, std::bind(&Material::ResourceNotify, this))
		, m_roughness_map(con_struct.roughness_map, std::bind(&Material::ResourceNotify, this))
		, m_emission_map(con_struct.emission_map, std::bind(&Material::ResourceNotify, this))
	{
		SetColor(con_struct.color);
		SetMetalic(con_struct.metalic);
		SetSpecular(con_struct.specular);
		SetRoughness(con_struct.roughness);
		SetEmission(con_struct.emission);
		SetIOR(con_struct.ior);
		SetScattering(con_struct.scattering);
	}

	void Material::SetColor(const Graphics::Color& color)
	{
		m_color = color;
		GetStateRegister().MakeModified();
	}
	void Material::SetMetalic(const float& metalic)
	{
		m_metalic = std::clamp(metalic, 0.0f, 1.0f);
		GetStateRegister().MakeModified();
	}
	void Material::SetSpecular(const float& specular)
	{
		m_specular = std::clamp(specular, 0.0f, 1.0f);
		GetStateRegister().MakeModified();
	}
	void Material::SetRoughness(const float& roughness)
	{
		m_roughness = std::clamp(roughness, 0.0f, 1.0f);
		GetStateRegister().MakeModified();
	}
	void Material::SetEmission(const float& emission)
	{
		m_emission = std::max(emission, 0.0f);
		GetStateRegister().MakeModified();
	}
	void Material::SetIOR(const float& ior)
	{
		m_ior = std::max(ior, 1.0f);
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
	void Material::SetNormalMap(const Handle<NormalMap>& normal_map)
	{
		m_normal_map = normal_map;
		GetStateRegister().MakeModified();
	}
	void Material::SetMetalicMap(const Handle<MetalicMap>& metalic_map)
	{
		m_metalic_map = metalic_map;
		GetStateRegister().MakeModified();
	}
	void Material::SetSpecularMap(const Handle<SpecularMap>& specular_map)
	{
		m_specular_map = specular_map;
		GetStateRegister().MakeModified();
	}
	void Material::SetRoughnessMap(const Handle<RoughnessMap>& roughness_map)
	{
		m_roughness_map = roughness_map;
		GetStateRegister().MakeModified();
	}
	void Material::SetEmissionMap(const Handle<EmissionMap>& emission_map)
	{
		m_emission_map = emission_map;
		GetStateRegister().MakeModified();
	}


	const Graphics::Color& Material::GetColor() const noexcept
	{
		return m_color;
	}
	float Material::GetMetalic() const noexcept
	{
		return m_metalic;
	}
	float Material::GetSpecular() const noexcept
	{
		return m_specular;
	}
	float Material::GetRoughness() const noexcept
	{
		return m_roughness;
	}
	float Material::GetEmission() const noexcept
	{
		return m_emission;
	}
	float Material::GetIOR() const noexcept
	{
		return m_ior;
	}
	float Material::GetScattering() const noexcept
	{
		return m_scattering;
	}

	const Handle<Texture>& Material::GetTexture() const
	{
		return static_cast<const Handle<Texture>&>(m_texture);
	}
	const Handle<NormalMap>& Material::GetNormalMap() const
	{
		return static_cast<const Handle<NormalMap>&>(m_normal_map);
	}
	const Handle<MetalicMap>& Material::GetMetalicMap() const
	{
		return static_cast<const Handle<MetalicMap>&>(m_metalic_map);
	}
	const Handle<SpecularMap>& Material::GetSpecularMap() const
	{
		return static_cast<const Handle<SpecularMap>&>(m_specular_map);
	}
	const Handle<RoughnessMap>& Material::GetRoughnessMap() const
	{
		return static_cast<const Handle<RoughnessMap>&>(m_roughness_map);
	}
	const Handle<EmissionMap>& Material::GetEmissionMap() const
	{
		return static_cast<const Handle<EmissionMap>&>(m_emission_map);
	}

	void Material::ResourceNotify()
	{
		GetStateRegister().MakeModified();
	}
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
}