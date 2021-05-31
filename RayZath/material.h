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

		float m_reflectance;
		float m_glossiness;

		float m_transmittance;
		float m_ior;

		float m_emittance;
		float m_scattering;

		Observer<Texture> m_texture;
		Observer<NormalMap> m_normal_map;
		Observer<EmittanceMap> m_emittance_map;
		Observer<ReflectanceMap> m_reflection_map;


	public:
		Material(
			Updatable* updatable, 
			const ConStruct<Material>& con_struct);
		Material(const Material& material) = delete;
		Material(Material&& material) = delete;
		~Material();


	public:
		Material& operator=(const Material& material) = delete;
		Material& operator=(Material&& material) = delete;


	public:
		void SetColor(const Graphics::Color& color);
		void SetReflectance(const float& reflectance);
		void SetGlossiness(const float& glossiness);
		void SetTransmittance(const float& transmittance);
		void SetIndexOfRefraction(const float& ior);
		void SetEmittance(const float& emittance);
		void SetScattering(const float& scattering);

		void SetTexture(const Handle<Texture>& texture);
		void SetNormalMap(const Handle<NormalMap>& normal_map);
		void SetEmittanceMap(const Handle<EmittanceMap>& emittance_map);
		void SetReflectanceMap(const Handle<ReflectanceMap>& reflectance_map);

		const Graphics::Color& GetColor() const noexcept;
		float GetReflectance() const noexcept;
		float GetGlossiness() const noexcept;
		float GetTransmittance() const noexcept;
		float GetIndexOfRefraction() const noexcept;
		float GetEmittance() const noexcept;
		float GetScattering() const noexcept;

		const Handle<Texture>& GetTexture() const;
		const Handle<NormalMap>& GetNormalMap() const;
		const Handle<EmittanceMap>& GetEmittanceMap() const;
		const Handle<ReflectanceMap>& GetReflectanceMap() const;
	private:
		void ResourceNotify();
	};

	template<> struct ConStruct<Material> : public ConStruct<WorldObject>
	{
		Graphics::Color color;

		float reflectance;
		float glossiness;

		float transmittance;
		float ior;

		float emittance;
		float scattering;

		Handle<Texture> texture;
		Handle<NormalMap> normal_map;
		Handle<EmittanceMap> emittance_map;
		Handle<ReflectanceMap> reflectance_map;


		ConStruct(
			const Graphics::Color& color = Graphics::Color::Palette::LightGrey,
			const float& reflectance = 0.0f,
			const float& glossiness = 0.0f,
			const float& transmittance = 0.0f,
			const float& ior = 1.0f,
			const float& emittance = 0.0f,
			const float& scattering = 0.0f,
			const Handle<Texture>& texture = Handle<Texture>(),
			const Handle<NormalMap>& normal_map = Handle<NormalMap>(),
			const Handle<EmittanceMap>& emittance_map = Handle<EmittanceMap>(),
			const Handle<ReflectanceMap>& reflectance_map = Handle<ReflectanceMap>())
			: color(color)
			, reflectance(reflectance)
			, glossiness(glossiness)
			, transmittance(transmittance)
			, ior(ior)
			, emittance(emittance)
			, scattering(scattering)
			, texture(texture)
			, normal_map(normal_map)
			, emittance_map(emittance_map)
			, reflectance_map(reflectance_map)
		{}
	};
}

#endif // !MATERIAL_H