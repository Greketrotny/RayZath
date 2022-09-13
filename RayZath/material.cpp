#include "material.hpp"

#include <algorithm>
#include <fstream>
#include <sstream>
#include <string>

namespace RayZath::Engine
{
	// ~~~~~~~~ [STRUCT] Material ~~~~~~~~
	Material::Material(
		Updatable* updatable,
		const ConStruct<Material>& con_struct)
		: WorldObject(updatable, con_struct)
		, m_texture(con_struct.texture, std::bind(&Material::resourceNotify, this))
		, m_normal_map(con_struct.normal_map, std::bind(&Material::resourceNotify, this))
		, m_metalness_map(con_struct.metalness_map, std::bind(&Material::resourceNotify, this))
		, m_roughness_map(con_struct.roughness_map, std::bind(&Material::resourceNotify, this))
		, m_emission_map(con_struct.emission_map, std::bind(&Material::resourceNotify, this))
	{
		color(con_struct.color);
		metalness(con_struct.metalness);
		roughness(con_struct.roughness);
		emission(con_struct.emission);
		ior(con_struct.ior);
		scattering(con_struct.scattering);
	}

	void Material::color(const Graphics::Color& color)
	{
		m_color = color;
		stateRegister().MakeModified();
	}
	void Material::metalness(const float& metalness)
	{
		m_metalness = std::clamp(metalness, 0.0f, 1.0f);
		stateRegister().MakeModified();
	}
	void Material::roughness(const float& roughness)
	{
		m_roughness = std::clamp(roughness, 0.0f, 1.0f);
		stateRegister().MakeModified();
	}
	void Material::emission(const float& emission)
	{
		m_emission = std::max(emission, 0.0f);
		stateRegister().MakeModified();
	}
	void Material::ior(const float& ior)
	{
		m_ior = std::max(ior, 1.0f);
		stateRegister().MakeModified();
	}
	void Material::scattering(const float& scattering)
	{
		m_scattering = std::max(0.0f, scattering);
		stateRegister().MakeModified();
	}

	void Material::texture(const Handle<Texture>& texture)
	{
		m_texture = texture;
		stateRegister().MakeModified();
	}
	void Material::normalMap(const Handle<NormalMap>& normal_map)
	{
		m_normal_map = normal_map;
		stateRegister().MakeModified();
	}
	void Material::metalnessMap(const Handle<MetalnessMap>& metalness_map)
	{
		m_metalness_map = metalness_map;
		stateRegister().MakeModified();
	}
	void Material::roughnessMap(const Handle<RoughnessMap>& roughness_map)
	{
		m_roughness_map = roughness_map;
		stateRegister().MakeModified();
	}
	void Material::emissionMap(const Handle<EmissionMap>& emission_map)
	{
		m_emission_map = emission_map;
		stateRegister().MakeModified();
	}

	const Graphics::Color& Material::color() const noexcept
	{
		return m_color;
	}
	float Material::metalness() const noexcept
	{
		return m_metalness;
	}
	float Material::roughness() const noexcept
	{
		return m_roughness;
	}
	float Material::emission() const noexcept
	{
		return m_emission;
	}
	float Material::ior() const noexcept
	{
		return m_ior;
	}
	float Material::scattering() const noexcept
	{
		return m_scattering;
	}

	const Handle<Texture>& Material::texture() const
	{
		return static_cast<const Handle<Texture>&>(m_texture);
	}
	const Handle<NormalMap>& Material::normalMap() const
	{
		return static_cast<const Handle<NormalMap>&>(m_normal_map);
	}
	const Handle<MetalnessMap>& Material::metalnessMap() const
	{
		return static_cast<const Handle<MetalnessMap>&>(m_metalness_map);
	}
	const Handle<RoughnessMap>& Material::roughnessMap() const
	{
		return static_cast<const Handle<RoughnessMap>&>(m_roughness_map);
	}
	const Handle<EmissionMap>& Material::emissionMap() const
	{
		return static_cast<const Handle<EmissionMap>&>(m_emission_map);
	}

	void Material::resourceNotify()
	{
		stateRegister().MakeModified();
	}

	template<>
	ConStruct<Material> Material::generateMaterial<Material::Common::Gold>()
	{
		return ConStruct<Material>(
			"generated_gold",
			Graphics::Color(0xFF, 0xD7, 0x00, 0xFF),
			1.0f, 0.001f, 0.0f, 1.0f, 0.0f);
	}
	template<>
	ConStruct<Material> Material::generateMaterial<Material::Common::Silver>()
	{
		return ConStruct<Material>(
			"generated_silver",
			Graphics::Color(0xC0, 0xC0, 0xC0, 0xFF),
			1.0f, 0.001f, 0.0f, 1.0f, 0.0f);
	}
	template<>
	ConStruct<Material> Material::generateMaterial<Material::Common::Copper>()
	{
		return ConStruct<Material>(
			"generated_copper",
			Graphics::Color(0xB8, 0x73, 0x33, 0xFF),
			1.0f, 0.001f, 0.0f, 1.0f, 0.0f);
	}

	template<>
	ConStruct<Material> Material::generateMaterial<Material::Common::Glass>()
	{
		return ConStruct<Material>(
			"generated_glass",
			Graphics::Color(0xFF, 0xFF, 0xFF, 0x00),
			0.0f, 0.0f, 0.0f, 1.45f, 0.0f);
	}
	template<>
	ConStruct<Material> Material::generateMaterial<Material::Common::Water>()
	{
		return ConStruct<Material>(
			"generated_water",
			Graphics::Color(0xFF, 0xFF, 0xFF, 0x00),
			0.0f, 0.0f, 0.0f, 1.33f, 0.0f);
	}
	template<>
	ConStruct<Material> Material::generateMaterial<Material::Common::Mirror>()
	{
		return ConStruct<Material>(
			"generated_mirror",
			Graphics::Color(0xF0, 0xF0, 0xF0, 0xFF),
			0.9f, 0.0f, 0.0f, 1.0f, 0.0f);
	}

	template<>
	ConStruct<Material> Material::generateMaterial<Material::Common::RoughWood>()
	{
		return ConStruct<Material>(
			"generated_rough_wood",
			Graphics::Color(0x96, 0x6F, 0x33, 0xFF),
			0.0f, 0.1f, 0.0f, 1.5f, 0.0f);
	}
	template<>
	ConStruct<Material> Material::generateMaterial<Material::Common::PolishedWood>()
	{
		return ConStruct<Material>(
			"generated_polished_wood",
			Graphics::Color(0x96, 0x6F, 0x33, 0xFF),
			0.0f, 0.002f, 0.0f, 1.5f, 0.0f);
	}

	template<>
	ConStruct<Material> Material::generateMaterial<Material::Common::Paper>()
	{
		return ConStruct<Material>(
			"generated_paper",
			Graphics::Color::Palette::White,
			0.0f, 0.0f, 0.0f, 1.0f, 0.0f);
	}
	template<>
	ConStruct<Material> Material::generateMaterial<Material::Common::Rubber>()
	{
		return ConStruct<Material>(
			"generated_rubber",
			Graphics::Color::Palette::Black,
			0.0f, 0.018f, 0.0f, 1.3f, 0.0f);
	}
	template<>
	ConStruct<Material> Material::generateMaterial<Material::Common::RoughPlastic>()
	{
		return ConStruct<Material>(
			"generated_rough_plastic",
			Graphics::Color::Palette::White,
			0.0f, 0.45f, 0.0f, 1.5f, 0.0f);
	}
	template<>
	ConStruct<Material> Material::generateMaterial<Material::Common::PolishedPlastic>()
	{
		return ConStruct<Material>(
			"generated_polished_plastic",
			Graphics::Color::Palette::White,
			0.0f, 0.0015f, 0.0f, 1.5f, 0.0f);
	}
	template<>
	ConStruct<Material> Material::generateMaterial<Material::Common::Porcelain>()
	{
		return ConStruct<Material>(
			"generated_porcelain",
			Graphics::Color::Palette::White,
			0.0f, 0.0f, 0.0f, 1.5f, 0.0f);
	}

	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
}