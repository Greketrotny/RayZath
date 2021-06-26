#include "world.h"

#include <string_view>
#include <fstream>
#include <sstream>

namespace RayZath
{
	// ~~~~~~~~ [CLASS] World ~~~~~~~~
	World::World(
		const uint32_t& maxCamerasCount,
		const uint32_t& maxLightsCount,
		const uint32_t& maxRenderObjectsCount)
		: Updatable(nullptr)
		, m_containers(
			ObjectContainer<Texture>(this, 64u),
			ObjectContainer<NormalMap>(this, 64u),
			ObjectContainer<MetalnessMap>(this, 64u),
			ObjectContainer<SpecularityMap>(this, 64u),
			ObjectContainer<RoughnessMap>(this, 64u),
			ObjectContainer<EmissionMap>(this, 64u),
			ObjectContainer<Material>(this, 64u),
			ObjectContainer<MeshStructure>(this, 1024u),
			ObjectContainer<Camera>(this, maxCamerasCount),
			ObjectContainer<PointLight>(this, maxLightsCount),
			ObjectContainer<SpotLight>(this, maxLightsCount),
			ObjectContainer<DirectLight>(this, maxLightsCount),
			ObjectContainerWithBVH<Mesh>(this, maxRenderObjectsCount),
			ObjectContainerWithBVH<Sphere>(this, maxRenderObjectsCount),
			ObjectContainer<Plane>(this, maxRenderObjectsCount))
		, m_material(
			this,
			ConStruct<Material>(
				Graphics::Color(0xFF, 0xFF, 0xFF, 0x00),
				0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f)),
		m_default_material(
			this, 
			ConStruct<Material>(
				Graphics::Color::Palette::LightGrey))
	{}

	Material& World::GetMaterial()
	{
		return m_material;
	}
	const Material& World::GetMaterial() const
	{
		return m_material;
	}
	Material& World::GetDefaultMaterial()
	{
		return m_default_material;
	}
	const Material& World::GetDefaultMaterial() const
	{
		return m_default_material;
	}

	std::tuple<std::string, std::string, std::string> ParseFileName(const std::string& file_name)
	{
		// extract file extension
		const size_t dot_idx = file_name.find_last_of('.');
		if (dot_idx == std::string::npos) return {};	// no dot in the file_name (invalid name)
		const std::string extension = file_name.substr(dot_idx + 1ull, file_name.size() - dot_idx);

		// extract file path
		const std::string path_name = file_name.substr(0ull, dot_idx);
		size_t last_slash_idx = path_name.find_last_of('/');
		if (last_slash_idx == std::string::npos)
			last_slash_idx = 0ull;
		const std::string path = file_name.substr(0ull, last_slash_idx);

		// extract file name
		const std::string name = file_name.substr(last_slash_idx + 1ull, dot_idx - last_slash_idx - 1ull);

		return std::tie(path, name, extension);
	}
	std::vector<Handle<Material>> World::LoadMTL(const std::string& full_file_name)
	{
		// decompose file extension
		auto [path, file_name, extension] = ParseFileName(full_file_name);
		if (extension != "mtl")
			throw RayZath::Exception(
				"File \"" + file_name + "." + extension + 
				"\" could not be recognized as valid Material Template Library (.mtl) file.");

		// open specified file
		std::ifstream ifs(full_file_name, std::ios_base::in);
		if (!ifs.is_open())
			throw RayZath::Exception(
				"Failed to open file " + full_file_name);

		auto trim_spaces = [](std::string& s)
		{
			const size_t first = s.find_first_not_of(' ');
			if (first == std::string::npos) return;

			const size_t last = s.find_last_not_of(' ');
			s = s.substr(first, (last - first + 1));
		};

		std::vector<Handle<Material>> materials;
		Handle<Material> material;


		// [>] Search for first "newmtl" keyword
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
					std::string material_name;
					ss >> material_name;
					material = this->Container<ContainerType::Material>().Create({});
					if (!material)
						throw RayZath::Exception(
							"Failed to create new material during MTL parsing.");

					materials.push_back(material);
					break;
				}
			}
		}

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
				{	// begin to read properties for new material

					material = this->Container<ContainerType::Material>().Create({});
					if (!material)
						throw RayZath::Exception(
							"Failed to create new material during MTL parsing.");
					materials.push_back(material);
				}
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

					material->SetColor(color);
				}
				else if (property == "Ks")
				{	// material specularity
					// for simplicity only the first value from three color channels 
					// is taken for consideration as specularity factor

					float specularity = 0.0f;
					ss >> specularity;
					material->SetSpecularity(specularity);
				}
				else if (property == "Ns")
				{	// roughness

					float exponent = 0.0f;
					ss >> exponent;

					const float exponent_max = 1000.0f;
					exponent = std::clamp(exponent, std::numeric_limits<float>::epsilon(), exponent_max);
					const float roughness = 1.0f - (log10f(exponent) / log10f(exponent_max));
					material->SetRoughness(roughness);
				}
				else if (property == "d")
				{	// dissolve/opaque (1 - transparency)

					float dissolve = 1.0f;
					ss >> dissolve;
					dissolve = std::clamp(dissolve, 0.0f, 1.0f);

					Graphics::Color color = material->GetColor();
					color.alpha = uint8_t(dissolve * 255.0f);
					material->SetColor(color);
				}
				else if (property == "Tr")
				{	// transparency

					float tr = 0.0f;
					ss >> tr;
					tr = std::clamp(tr, 0.0f, 1.0f);

					Graphics::Color color = material->GetColor();
					color.alpha = uint8_t((1.0f - tr) * 255.0f);
					material->SetColor(color);
				}
				else if (property == "Ni")
				{	// IOR

					float ior = 1.0f;
					ss >> ior;
					material->SetIOR(ior);
				}

				// extended (PBR) material properties
				else if (property == "Pm")
				{
					float metallic = 0.0f;
					ss >> metallic;
					material->SetMetalness(metallic);
				}
				else if (property == "Pr")
				{
					float roughness = 0.0f;
					ss >> roughness;
					material->SetRoughness(roughness);
				}
				else if (property == "Ke")
				{
					float emission;
					ss >> emission;
					material->SetEmission(emission);
				}
			}
		}


		ifs.close();

		return materials;
	}

	void World::DestroyAll()
	{
		Container<ContainerType::Texture>().DestroyAll();
		Container<ContainerType::NormalMap>().DestroyAll();
		Container<ContainerType::MetalnessMap>().DestroyAll();
		Container<ContainerType::SpecularityMap>().DestroyAll();
		Container<ContainerType::RoughnessMap>().DestroyAll();
		Container<ContainerType::EmissionMap>().DestroyAll();

		Container<ContainerType::Material>().DestroyAll();
		Container<ContainerType::MeshStructure>().DestroyAll();

		Container<ContainerType::Camera>().DestroyAll();

		Container<ContainerType::PointLight>().DestroyAll();
		Container<ContainerType::SpotLight>().DestroyAll();
		Container<ContainerType::DirectLight>().DestroyAll();

		Container<ContainerType::Mesh>().DestroyAll();
		Container<ContainerType::Sphere>().DestroyAll();
		Container<ContainerType::Plane>().DestroyAll();
	}

	void World::Update()
	{
		if (!GetStateRegister().RequiresUpdate()) return;

		
		Container<ContainerType::Texture>().Update();
		Container<ContainerType::NormalMap>().Update();
		Container<ContainerType::MetalnessMap>().Update();
		Container<ContainerType::SpecularityMap>().Update();
		Container<ContainerType::RoughnessMap>().Update();
		Container<ContainerType::EmissionMap>().Update();

		Container<ContainerType::Material>().Update();
		Container<ContainerType::MeshStructure>().Update();

		Container<ContainerType::Camera>().Update();

		Container<ContainerType::PointLight>().Update();
		Container<ContainerType::SpotLight>().Update();
		Container<ContainerType::DirectLight>().Update();

		Container<ContainerType::Mesh>().Update();
		Container<ContainerType::Sphere>().Update();
		Container<ContainerType::Plane>().Update();

		GetStateRegister().Update();
	}
}