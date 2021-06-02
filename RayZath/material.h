#ifndef MATERIAL_H
#define MATERIAL_H

#include "world_object.h"
#include "render_parts.h"

namespace RayZath
{
	struct Material
		: public WorldObject
	{
	private:
		Graphics::Color m_color;

		float m_metalic;
		float m_specular;
		float m_roughness;
		float m_emission;

		float m_transmission;
		float m_ior;
		float m_scattering;

		Observer<Texture> m_texture;
		Observer<NormalMap> m_normal_map;
		Observer<MetalicMap> m_metalic_map;
		Observer<SpecularMap> m_specular_map;
		Observer<RoughnessMap> m_roughness_map;
		Observer<EmissionMap> m_emission_map;


	public:
		Material(
			Updatable* updatable, 
			const ConStruct<Material>& con_struct);
		Material(const Material& material) = delete;
		Material(Material&& material) = delete;


	public:
		Material& operator=(const Material& material) = delete;
		Material& operator=(Material&& material) = delete;


	public:
		void SetColor(const Graphics::Color& color);
		void SetMetalic(const float& metalic);
		void SetSpecular(const float& specular);
		void SetRoughness(const float& roughness);
		void SetEmission(const float& emission);
		void SetTransmission(const float& transmission);
		void SetIOR(const float& ior);
		void SetScattering(const float& scattering);

		void SetTexture(const Handle<Texture>& texture);
		void SetNormalMap(const Handle<NormalMap>& normal_map);
		void SetMetalicMap(const Handle<MetalicMap>& metalic_map);
		void SetSpecularMap(const Handle<SpecularMap>& specular_map);
		void SetRoughnessMap(const Handle<RoughnessMap>& roughness_map);
		void SetEmissionMap(const Handle<EmissionMap>& emission_map);


		const Graphics::Color& GetColor() const noexcept;
		float GetMetalic() const noexcept;
		float GetSpecular() const noexcept;
		float GetRoughness() const noexcept;
		float GetEmission() const noexcept;
		float GetTransmission() const noexcept;
		float GetIOR() const noexcept;
		float GetScattering() const noexcept;

		const Handle<Texture>& GetTexture() const;
		const Handle<NormalMap>& GetNormalMap() const;
		const Handle<MetalicMap>& GetMetalicMap() const;
		const Handle<SpecularMap>& GetSpecularMap() const;
		const Handle<RoughnessMap>& GetRoughnessMap() const;
		const Handle<EmissionMap>& GetEmissionMap() const;
	private:
		void ResourceNotify();
	};

	template<> struct ConStruct<Material> : public ConStruct<WorldObject>
	{
		Graphics::Color color;

		float metalic;
		float specular;
		float roughness;
		float emission;

		float transmission;
		float ior;
		float scattering;

		Handle<Texture> texture;
		Handle<NormalMap> normal_map;
		Handle<MetalicMap> metalic_map;
		Handle<SpecularMap> specular_map;
		Handle<RoughnessMap> roughness_map;
		Handle<EmissionMap> emission_map;


		ConStruct(
			const Graphics::Color& color = Graphics::Color::Palette::LightGrey,
			const float& metalic = 0.0f,
			const float& specular = 0.0f,
			const float& roughness = 0.0f,
			const float& emission = 0.0f,
			const float& transmission = 0.0f,
			const float& ior = 1.0f,
			const float& scattering = 0.0f,
			const Handle<Texture>& texture = Handle<Texture>(),
			const Handle<NormalMap>& normal_map = Handle<NormalMap>(),
			const Handle<MetalicMap>& metalic_map = Handle<MetalicMap>(),
			const Handle<SpecularMap>& specular_map = Handle<SpecularMap>(),
			const Handle<RoughnessMap>& roughness_map = Handle<RoughnessMap>(),
			const Handle<EmissionMap>& emission_map = Handle<EmissionMap>())
			: color(color)
			, metalic(metalic)
			, specular(specular)
			, roughness(roughness)
			, emission(emission)
			, transmission(transmission)
			, ior(ior)
			, scattering(scattering)
			, texture(texture)
			, normal_map(normal_map)
			, metalic_map(metalic_map)
			, specular_map(specular_map)
			, roughness_map(roughness_map)
			, emission_map(emission_map)
		{}
	};
}

#endif // !MATERIAL_H