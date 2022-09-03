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
		, m_texture(con_struct.texture, std::bind(&Material::ResourceNotify, this))
		, m_normal_map(con_struct.normal_map, std::bind(&Material::ResourceNotify, this))
		, m_metalness_map(con_struct.metalness_map, std::bind(&Material::ResourceNotify, this))
		, m_roughness_map(con_struct.roughness_map, std::bind(&Material::ResourceNotify, this))
		, m_emission_map(con_struct.emission_map, std::bind(&Material::ResourceNotify, this))
	{
		SetColor(con_struct.color);
		SetMetalness(con_struct.metalness);
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
	void Material::SetMetalness(const float& metalness)
	{
		m_metalness = std::clamp(metalness, 0.0f, 1.0f);
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
	void Material::SetMetalnessMap(const Handle<MetalnessMap>& metalness_map)
	{
		m_metalness_map = metalness_map;
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
	float Material::GetMetalness() const noexcept
	{
		return m_metalness;
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
	const Handle<MetalnessMap>& Material::GetMetalnessMap() const
	{
		return static_cast<const Handle<MetalnessMap>&>(m_metalness_map);
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

	bool Material::LoadFromFile(const std::string& file_name)
	{
		// [>] Open specified file
		std::ifstream ifs;
		ifs.open(file_name, std::ios_base::in);
		if (!ifs.is_open()) return false;


		auto trim_spaces = [](std::string& s)
		{
			const size_t first = s.find_first_not_of(' ');
			if (first == std::string::npos) return;

			const size_t last = s.find_last_not_of(' ');
			s = s.substr(first, (last - first + 1));
		};


		// [>] Search for "newmtl" keyword
		{
			std::string file_line;
			while (std::getline(ifs, file_line))
			{
				trim_spaces(file_line);
				if (file_line.empty()) continue;

				std::stringstream ss(file_line);
				std::string newmtl;
				ss >> newmtl;

				if (newmtl == "newmtl")
				{
					std::string material_name = "loaded name";
					ss >> material_name;
					SetName(material_name);
					break;
				}
			}
		}


		// [>] Read material properties
		{
			std::string file_line;
			while (std::getline(ifs, file_line))
			{
				trim_spaces(file_line);
				if (file_line.empty()) continue;

				std::stringstream ss(file_line);
				std::string property;
				ss >> property;

				if (property == "newmtl")
				{	// attempt to read next material definition -> return
					break;
				}

				// standard .mtl properties
				else if (property == "Kd")
				{	// material color

					// collect diffuse color values
					std::vector<float> values;
					while (!ss.eof())
					{
						float value;
						ss >> value;
						value = std::clamp(value, 0.0f, 1.0f);
						values.push_back(value);
					}

					// set material color
					Graphics::Color color = Graphics::Color::Palette::Grey;
					if (values.size() >= 3ull)
						color = Graphics::Color(
							uint8_t(values[0] * 255.0f),
							uint8_t(values[1] * 255.0f),
							uint8_t(values[2] * 255.0f));
					else if (values.size() == 1ull)
						color = Graphics::Color(uint8_t(values[0] * 255.0f));

					SetColor(color);
				}
				else if (property == "Ns")
				{	// roughness

					float exponent = 0.0f;
					ss >> exponent;

					const float exponent_max = 1000.0f;
					exponent = std::clamp(exponent, std::numeric_limits<float>::epsilon(), exponent_max);
					const float roughness = 1.0f - (log10f(exponent) / log10f(exponent_max));
					SetRoughness(roughness);
				}
				else if (property == "d")
				{	// dissolve/opaque (1 - transparency)

					float dissolve = 1.0f;
					ss >> dissolve;
					dissolve = std::clamp(dissolve, 0.0f, 1.0f);

					Graphics::Color color = GetColor();
					color.alpha = uint8_t(dissolve * 255.0f);
					SetColor(color);
				}
				else if (property == "Tr")
				{	// transparency

					float tr = 0.0f;
					ss >> tr;
					tr = std::clamp(tr, 0.0f, 1.0f);

					Graphics::Color color = GetColor();
					color.alpha = uint8_t((1.0f - tr) * 255.0f);
					SetColor(color);
				}
				else if (property == "Ni")
				{	// IOR

					float ior = 1.0f;
					ss >> ior;
					SetIOR(ior);
				}

				// extended (PBR) material properties
				else if (property == "Pm")
				{
					float metallic = 0.0f;
					ss >> metallic;
					SetMetalness(metallic);
				}
				else if (property == "Pr")
				{
					float roughness = 0.0f;
					ss >> roughness;
					SetRoughness(roughness);
				}
				else if (property == "Ke")
				{
					float emission;
					ss >> emission;
					SetEmission(emission);
				}
			}
		}

		ifs.close();
		return true;
	}

	template<>
	ConStruct<Material> Material::GenerateMaterial<Material::Common::Gold>()
	{
		return ConStruct<Material>(
			"generated_gold",
			Graphics::Color(0xFF, 0xD7, 0x00, 0xFF),
			1.0f, 0.001f, 0.0f, 1.0f, 0.0f);
	}
	template<>
	ConStruct<Material> Material::GenerateMaterial<Material::Common::Silver>()
	{
		return ConStruct<Material>(
			"generated_silver",
			Graphics::Color(0xC0, 0xC0, 0xC0, 0xFF),
			1.0f, 0.001f, 0.0f, 1.0f, 0.0f);
	}
	template<>
	ConStruct<Material> Material::GenerateMaterial<Material::Common::Copper>()
	{
		return ConStruct<Material>(
			"generated_copper",
			Graphics::Color(0xB8, 0x73, 0x33, 0xFF),
			1.0f, 0.001f, 0.0f, 1.0f, 0.0f);
	}

	template<>
	ConStruct<Material> Material::GenerateMaterial<Material::Common::Glass>()
	{
		return ConStruct<Material>(
			"generated_glass",
			Graphics::Color(0xFF, 0xFF, 0xFF, 0x00),
			0.0f, 0.0f, 0.0f, 1.45f, 0.0f);
	}
	template<>
	ConStruct<Material> Material::GenerateMaterial<Material::Common::Water>()
	{
		return ConStruct<Material>(
			"generated_water",
			Graphics::Color(0xFF, 0xFF, 0xFF, 0x00),
			0.0f, 0.0f, 0.0f, 1.33f, 0.0f);
	}
	template<>
	ConStruct<Material> Material::GenerateMaterial<Material::Common::Mirror>()
	{
		return ConStruct<Material>(
			"generated_mirror",
			Graphics::Color(0xF0, 0xF0, 0xF0, 0xFF),
			0.9f, 0.0f, 0.0f, 1.0f, 0.0f);
	}

	template<>
	ConStruct<Material> Material::GenerateMaterial<Material::Common::RoughWood>()
	{
		return ConStruct<Material>(
			"generated_rough_wood",
			Graphics::Color(0x96, 0x6F, 0x33, 0xFF),
			0.0f, 0.1f, 0.0f, 1.5f, 0.0f);
	}
	template<>
	ConStruct<Material> Material::GenerateMaterial<Material::Common::PolishedWood>()
	{
		return ConStruct<Material>(
			"generated_polished_wood",
			Graphics::Color(0x96, 0x6F, 0x33, 0xFF),
			0.0f, 0.002f, 0.0f, 1.5f, 0.0f);
	}

	template<>
	ConStruct<Material> Material::GenerateMaterial<Material::Common::Paper>()
	{
		return ConStruct<Material>(
			"generated_paper",
			Graphics::Color::Palette::White,
			0.0f, 0.0f, 0.0f, 1.0f, 0.0f);
	}
	template<>
	ConStruct<Material> Material::GenerateMaterial<Material::Common::Rubber>()
	{
		return ConStruct<Material>(
			"generated_rubber",
			Graphics::Color::Palette::Black,
			0.0f, 0.018f, 0.0f, 1.3f, 0.0f);
	}
	template<>
	ConStruct<Material> Material::GenerateMaterial<Material::Common::RoughPlastic>()
	{
		return ConStruct<Material>(
			"generated_rough_plastic",
			Graphics::Color::Palette::White,
			0.0f, 0.45f, 0.0f, 1.5f, 0.0f);
	}
	template<>
	ConStruct<Material> Material::GenerateMaterial<Material::Common::PolishedPlastic>()
	{
		return ConStruct<Material>(
			"generated_polished_plastic",
			Graphics::Color::Palette::White,
			0.0f, 0.0015f, 0.0f, 1.5f, 0.0f);
	}
	template<>
	ConStruct<Material> Material::GenerateMaterial<Material::Common::Porcelain>()
	{
		return ConStruct<Material>(
			"generated_porcelain",
			Graphics::Color::Palette::White,
			0.0f, 0.0f, 0.0f, 1.5f, 0.0f);
	}

	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
}