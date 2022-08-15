#include "loader.hpp"

#include "json_loader.hpp"
#include "world.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include "./lib/stb_image/stb_image.h"

#include <fstream>
#include <sstream>
#include <string>
#include <memory>

namespace RayZath::Engine
{
	// ~~~~~~~~ LoaderBase ~~~~~~~~
	LoaderBase::LoaderBase(World& world)
		: mr_world(world)
	{}
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~


	// ~~~~~~~~ BitmapLoader ~~~~~~~~
	BitmapLoader::BitmapLoader(World& world)
		: LoaderBase(world)
	{}

	template <>
	Graphics::Bitmap BitmapLoader::LoadMap<World::ObjectType::Texture>(const std::string& path)
	{
		int width{}, height{}, components{};
		std::unique_ptr<stbi_uc, decltype(&stbi_image_free)> data(
			stbi_load(path.c_str(), &width, &height, &components, 4),
			&stbi_image_free);
		RZAssert(data, "failed to open texture");
		RZAssert(width != 0 && height != 0, "one dimension had size of 0");

		Graphics::Bitmap image(width, height, {});
		std::memcpy((void*)image.GetMapAddress(), data.get(),
			image.GetWidth() * image.GetHeight() * sizeof(*image.GetMapAddress()));
		return image;
	}
	template <>
	Graphics::Bitmap BitmapLoader::LoadMap<World::ObjectType::NormalMap>(const std::string& path)
	{
		return LoadMap<World::ObjectType::Texture>(path);
	}
	template <>
	Graphics::Buffer2D<uint8_t> BitmapLoader::LoadMap<World::ObjectType::MetalnessMap>(const std::string& path)
	{
		int width{}, height{}, components{};

		std::unique_ptr<stbi_uc, decltype(&stbi_image_free)> data(
			stbi_load(path.c_str(), &width, &height, &components, 1),
			&stbi_image_free);
		RZAssert(data, "failed to open texture");
		RZAssert(width != 0 && height != 0, "one dimension had size of 0");

		Graphics::Buffer2D<uint8_t> image(width, height, {});
		std::memcpy((void*)image.GetMapAddress(), data.get(),
			image.GetWidth() * image.GetHeight() * sizeof(*image.GetMapAddress()));
		return image;
	}
	template <>
	Graphics::Buffer2D<uint8_t> BitmapLoader::LoadMap<World::ObjectType::RoughnessMap>(const std::string& path)
	{
		return LoadMap<World::ObjectType::MetalnessMap>(path);
	}
	template <>
	Graphics::Buffer2D<float> BitmapLoader::LoadMap<World::ObjectType::EmissionMap>(const std::string& path)
	{
		int width{}, height{}, components{};
		std::unique_ptr<float, decltype(&stbi_image_free)> data(
			stbi_loadf(path.c_str(), &width, &height, &components, 1),
			&stbi_image_free);
		RZAssert(data, "failed to open emission map");
		RZAssert(width != 0 && height != 0, "one dimension had size of 0");

		Graphics::Buffer2D<float> image(width, height, {});
		std::memcpy((void*)image.GetMapAddress(), data.get(),
			image.GetWidth() * image.GetHeight() * sizeof(*image.GetMapAddress()));
		return image;
	}
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



	// ~~~~~~~~ MTLLoader ~~~~~~~~
	MTLLoader::MTLLoader(World& world)
		: BitmapLoader(world)
	{}

	std::vector<Handle<Material>> MTLLoader::LoadMTL(const std::filesystem::path& path)
	{
		/*
		* - material name only with underscores "material_name"
		* - color is a 3xfloat, when one float, the rest is assumed to be the same
		* - Tf, transmission filter
		*/
		RZAssert(path.has_filename(), path.string() + " doesn't contain file name");
		RZAssert(path.has_extension() && path.extension() == ".mtl", path.string() + " doesn't have .mtl extension");

		// open .mtl file
		std::ifstream ifs(path, std::ios_base::in);
		RZAssert(ifs.is_open(), "failed to open file " + path.string());

		auto trim_spaces = [](std::string& s)
		{
			const size_t first = s.find_first_not_of(' ');
			if (first == std::string::npos) return;

			const size_t last = s.find_last_not_of(' ');
			s = s.substr(first, (last - first + 1));
		};

		std::vector<Handle<Material>> loaded_materials;
		Handle<Material> material;

		std::vector<Handle<Texture>> loaded_textures;
		std::vector<Handle<NormalMap>> loaded_normal_maps;
		std::vector<Handle<MetalnessMap>> loaded_metalness_maps;
		std::vector<Handle<RoughnessMap>> loaded_roughness_maps;


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
					std::getline(ss, material_name);
					trim_spaces(material_name);
					material = mr_world.Container<World::ObjectType::Material>().Create(
						ConStruct<Material>(material_name));
					if (!material)
						throw RayZath::Exception(
							"Failed to create new material during MTL parsing.");

					loaded_materials.push_back(material);
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
				std::string parameter;
				ss >> parameter;

				if (parameter == "newmtl")
				{	// begin to read properties for new material

					std::string material_name;
					std::getline(ss, material_name);
					trim_spaces(material_name);
					material = mr_world.Container<World::ObjectType::Material>().Create(
						ConStruct<Material>(material_name));
					if (!material)
						throw RayZath::Exception(
							"Failed to create new material during MTL parsing.");
					loaded_materials.push_back(material);
				}
				else if (parameter == "Kd")
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
					Graphics::Color color = material->GetColor();
					if (values.size() >= 3ull)
						color = Graphics::Color(
							uint8_t(values[0] * 255.0f),
							uint8_t(values[1] * 255.0f),
							uint8_t(values[2] * 255.0f),
							color.alpha);
					else if (values.size() == 1ull)
						color = Graphics::Color(
							uint8_t(values[0] * 255.0f),
							uint8_t(values[0] * 255.0f),
							uint8_t(values[0] * 255.0f),
							color.alpha);

					material->SetColor(color);
				}
				else if (parameter == "Ns")
				{	// roughness

					float exponent = 0.0f;
					ss >> exponent;

					const float exponent_max = 1000.0f;
					exponent = std::clamp(exponent, std::numeric_limits<float>::epsilon(), exponent_max);
					const float roughness = 1.0f - (log10f(exponent) / log10f(exponent_max));
					material->SetRoughness(roughness);
				}
				else if (parameter == "d")
				{	// dissolve/opaque (1 - transparency)

					float dissolve = 1.0f;
					ss >> dissolve;
					dissolve = std::clamp(dissolve, 0.0f, 1.0f);

					Graphics::Color color = material->GetColor();
					color.alpha = uint8_t(dissolve * 255.0f);
					material->SetColor(color);
				}
				else if (parameter == "Tr")
				{	// transparency

					float tr = 0.0f;
					ss >> tr;
					tr = std::clamp(tr, 0.0f, 1.0f);

					Graphics::Color color = material->GetColor();
					color.alpha = uint8_t((1.0f - tr) * 255.0f);
					material->SetColor(color);
				}
				else if (parameter == "Ni")
				{	// IOR

					float ior = 1.0f;
					ss >> ior;
					material->SetIOR(ior);
				}

				// extended (PBR) material properties
				else if (parameter == "Pm")
				{
					float metallic = 0.0f;
					ss >> metallic;
					material->SetMetalness(metallic);
				}
				else if (parameter == "Pr")
				{
					float roughness = 0.0f;
					ss >> roughness;
					material->SetRoughness(roughness);
				}
				else if (parameter == "Ke")
				{
					float emission = 0.0f;
					ss >> emission;
					material->SetEmission(emission);
				}


				auto extract_map_path = [](const std::string& str_params) -> std::string
				{
					std::list<std::string> option_list;
					std::string path;

					std::stringstream ss(str_params);
					while (!ss.eof())
					{
						std::string str;
						ss >> str;
						option_list.push_back(str);
					}

					const std::unordered_map<std::string, int> n_values = {
						{"-bm", 1 },
						{"-blendu", 1 },
						{"-blendv", 1 },
						{"-boost", 1 },
						{"-cc", 1 },
						{"-clamp", 1 },
						{"-imfchan", 1 },
						{"-mm", 2 },
						{"-o", 3 },
						{"-s", 3 },
						{"-t", 3 },
						{"-texres", 1 } };

					std::list<std::string>::const_iterator curr_option = option_list.begin();
					while (!option_list.empty())
					{
						auto search = n_values.find(*curr_option);
						if (search != n_values.end())
						{
							auto values_iter = std::next(curr_option);
							for (int i = 0; i < search->second; i++)
							{
								if (values_iter != option_list.end())
									option_list.erase(values_iter++);
							}
						}
						else
						{
							path += *curr_option + " ";

						}
						option_list.erase(curr_option++);
					}

					if (path.back() == ' ') path.pop_back();
					return path;
				};

				// maps
				if (parameter == "map_Kd")
				{
					// extract texture path from parameter
					std::string map_string;
					std::getline(ss, map_string);
					trim_spaces(map_string);
					std::filesystem::path texture_path =
						extract_map_path(map_string);

					// search for already loaded texture with the same file name
					Handle<Texture> texture;
					for (auto& t : loaded_textures)
					{
						if (t->GetName() == texture_path.stem().string())
							texture = t;
					}

					if (texture)
					{	// texture with the name has been loaded - share texture
						material->SetTexture(texture);
					}
					else
					{	// texture hasn't been loaded yet - create new texture and load texture image
						if (texture_path.is_relative())
							texture_path = path.parent_path() / texture_path;
						texture = mr_world.Container<World::ObjectType::Texture>().Create(
							ConStruct<Texture>(texture_path.stem().string(),
								LoadMap<World::ObjectType::Texture>(texture_path.string())));
						loaded_textures.push_back(texture);
						material->SetTexture(texture);
					}
				}
				else if (parameter == "norm")
				{
					// extract normal map path from parameter
					std::string map_string;
					std::getline(ss, map_string);
					trim_spaces(map_string);
					std::filesystem::path normal_map_path =
						extract_map_path(map_string);

					// search for already loaded normal map with the same file name
					Handle<NormalMap> normal_map;
					for (auto& nm : loaded_normal_maps)
					{
						if (nm->GetName() == normal_map_path.stem().string())
							normal_map = nm;
					}

					if (normal_map)
					{	// normal_map with the name has been loaded - share normal_map
						material->SetNormalMap(normal_map);
					}
					else
					{	// normal_map hasn't been loaded yet - create new normal_map and load normal_map image
						if (normal_map_path.is_relative())
							normal_map_path = path.parent_path() / normal_map_path;
						normal_map = mr_world.Container<World::ObjectType::NormalMap>().Create(
							ConStruct<NormalMap>(normal_map_path.stem().string(),
								LoadMap<World::ObjectType::NormalMap>(normal_map_path.string())));
						loaded_normal_maps.push_back(normal_map);
						material->SetNormalMap(normal_map);
					}
				}
				else if (parameter == "map_Pm")
				{
					// extract metalness map path from parameter
					std::string map_string;
					std::getline(ss, map_string);
					trim_spaces(map_string);
					std::filesystem::path metalness_map_path =
						extract_map_path(map_string);

					// search for already loaded metalness map with the same file name
					Handle<MetalnessMap> metalness_map;
					for (auto& mm : loaded_metalness_maps)
					{
						if (mm->GetName() == metalness_map_path.stem().string())
							metalness_map = mm;
					}

					if (metalness_map)
					{	// metalness map with the name has been loaded - share metalness map
						material->SetMetalnessMap(metalness_map);
					}
					else
					{	// metalness map hasn't been loaded yet - create new metalness map and load metalness map image
						if (metalness_map_path.is_relative())
							metalness_map_path = path.parent_path() / metalness_map_path;
						metalness_map = mr_world.Container<World::ObjectType::MetalnessMap>().Create(
							ConStruct<MetalnessMap>(metalness_map_path.stem().string(),
								LoadMap<World::ObjectType::MetalnessMap>(metalness_map_path.string())));
						loaded_metalness_maps.push_back(metalness_map);
						material->SetMetalnessMap(metalness_map);
					}
				}
				else if (parameter == "map_Pr")
				{
					// extract roughness map path from parameter
					std::string map_string;
					std::getline(ss, map_string);
					trim_spaces(map_string);
					std::filesystem::path roughness_map_path =
						extract_map_path(map_string);

					// search for already loaded roughness map with the same file name
					Handle<RoughnessMap> roughness_map;
					for (auto& rm : loaded_roughness_maps)
					{
						if (rm->GetName() == roughness_map_path.stem().string())
							roughness_map = rm;
					}

					if (roughness_map)
					{	// roughness map with the name has been loaded - share roughness map
						material->SetRoughnessMap(roughness_map);
					}
					else
					{	// roughness map hasn't been loaded yet - create new roughness map and load roughness map image
						if (roughness_map_path.is_relative())
							roughness_map_path = path.parent_path() / roughness_map_path;
						roughness_map = mr_world.Container<World::ObjectType::RoughnessMap>().Create(
							ConStruct<RoughnessMap>(roughness_map_path.stem().string(),
								LoadMap<World::ObjectType::MetalnessMap>(roughness_map_path.string())));
						loaded_roughness_maps.push_back(roughness_map);
						material->SetRoughnessMap(roughness_map);
					}
				}
			}
		}

		ifs.close();

		return loaded_materials;
	}
	void MTLLoader::LoadMTL(const std::filesystem::path& path, Material& material)
	{
		RZAssert(path.has_filename(), path.string() + " doesn't contain file name");
		RZAssert(path.has_extension() && path.extension() == ".mtl", path.string() + " doesn't have .mtl extension");

		// open .mtl file
		std::ifstream ifs(path, std::ios_base::in);
		RZAssert(ifs.is_open(), "failed to open file " + path.string());

		auto trim_spaces = [](std::string& s)
		{
			const size_t first = s.find_first_not_of(' ');
			if (first == std::string::npos) return;

			const size_t last = s.find_last_not_of(' ');
			s = s.substr(first, (last - first + 1));
		};

		std::vector<Handle<Texture>> loaded_textures;
		std::vector<Handle<NormalMap>> loaded_normal_maps;
		std::vector<Handle<MetalnessMap>> loaded_metalness_maps;
		std::vector<Handle<RoughnessMap>> loaded_roughness_maps;

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
					std::getline(ss, material_name);
					trim_spaces(material_name);
					material.SetName(material_name);
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
				std::string parameter;
				ss >> parameter;

				if (parameter == "Kd")
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
					Graphics::Color color = material.GetColor();
					if (values.size() >= 3ull)
						color = Graphics::Color(
							uint8_t(values[0] * 255.0f),
							uint8_t(values[1] * 255.0f),
							uint8_t(values[2] * 255.0f),
							color.alpha);
					else if (values.size() == 1ull)
						color = Graphics::Color(
							uint8_t(values[0] * 255.0f),
							uint8_t(values[0] * 255.0f),
							uint8_t(values[0] * 255.0f),
							color.alpha);

					material.SetColor(color);
				}
				else if (parameter == "Ns")
				{	// roughness

					float exponent = 0.0f;
					ss >> exponent;

					const float exponent_max = 1000.0f;
					exponent = std::clamp(exponent, std::numeric_limits<float>::epsilon(), exponent_max);
					const float roughness = 1.0f - (log10f(exponent) / log10f(exponent_max));
					material.SetRoughness(roughness);
				}
				else if (parameter == "d")
				{	// dissolve/opaque (1 - transparency)

					float dissolve = 1.0f;
					ss >> dissolve;
					dissolve = std::clamp(dissolve, 0.0f, 1.0f);

					Graphics::Color color = material.GetColor();
					color.alpha = uint8_t(dissolve * 255.0f);
					material.SetColor(color);
				}
				else if (parameter == "Tr")
				{	// transparency

					float tr = 0.0f;
					ss >> tr;
					tr = std::clamp(tr, 0.0f, 1.0f);

					Graphics::Color color = material.GetColor();
					color.alpha = uint8_t((1.0f - tr) * 255.0f);
					material.SetColor(color);
				}
				else if (parameter == "Ni")
				{	// IOR

					float ior = 1.0f;
					ss >> ior;
					material.SetIOR(ior);
				}

				// extended (PBR) material properties
				else if (parameter == "Pm")
				{
					float metallic = 0.0f;
					ss >> metallic;
					material.SetMetalness(metallic);
				}
				else if (parameter == "Pr")
				{
					float roughness = 0.0f;
					ss >> roughness;
					material.SetRoughness(roughness);
				}
				else if (parameter == "Ke")
				{
					float emission = 0.0f;
					ss >> emission;
					material.SetEmission(emission);
				}


				auto extract_map_path = [](const std::string& str_params) -> std::string
				{
					std::list<std::string> option_list;
					std::string path;

					std::stringstream ss(str_params);
					while (!ss.eof())
					{
						std::string str;
						ss >> str;
						option_list.push_back(str);
					}

					const std::unordered_map<std::string, int> n_values = {
						{"-bm", 1 },
						{"-blendu", 1 },
						{"-blendv", 1 },
						{"-boost", 1 },
						{"-cc", 1 },
						{"-clamp", 1 },
						{"-imfchan", 1 },
						{"-mm", 2 },
						{"-o", 3 },
						{"-s", 3 },
						{"-t", 3 },
						{"-texres", 1 } };

					std::list<std::string>::const_iterator curr_option = option_list.begin();
					while (!option_list.empty())
					{
						auto search = n_values.find(*curr_option);
						if (search != n_values.end())
						{
							auto values_iter = std::next(curr_option);
							for (int i = 0; i < search->second; i++)
							{
								if (values_iter != option_list.end())
									option_list.erase(values_iter++);
							}
						}
						else
						{
							path += *curr_option + " ";

						}
						option_list.erase(curr_option++);
					}

					if (path.back() == ' ') path.pop_back();
					return path;
				};

				// maps
				if (parameter == "map_Kd")
				{
					// extract texture path from parameter
					std::string map_string;
					std::getline(ss, map_string);
					trim_spaces(map_string);
					std::filesystem::path texture_path =
						extract_map_path(map_string);

					// search for already loaded texture with the same file name
					Handle<Texture> texture;
					for (auto& t : loaded_textures)
					{
						if (t->GetName() == texture_path.stem().string())
							texture = t;
					}

					if (texture)
					{	// texture with the name has been loaded - share texture
						material.SetTexture(texture);
					}
					else
					{	// texture hasn't been loaded yet - create new texture and load texture image
						if (texture_path.is_relative())
							texture_path = path.parent_path() / texture_path;
						texture = mr_world.Container<World::ObjectType::Texture>().Create(
							ConStruct<Texture>(texture_path.stem().string(),
								LoadMap<World::ObjectType::Texture>(texture_path.string())));
						loaded_textures.push_back(texture);
						material.SetTexture(texture);
					}
				}
				else if (parameter == "norm")
				{
					// extract normal map path from parameter
					std::string map_string;
					std::getline(ss, map_string);
					trim_spaces(map_string);
					std::filesystem::path normal_map_path =
						extract_map_path(map_string);

					// search for already loaded normal map with the same file name
					Handle<NormalMap> normal_map;
					for (auto& nm : loaded_normal_maps)
					{
						if (nm->GetName() == normal_map_path.stem().string())
							normal_map = nm;
					}

					if (normal_map)
					{	// normal_map with the name has been loaded - share normal_map
						material.SetNormalMap(normal_map);
					}
					else
					{	// normal_map hasn't been loaded yet - create new normal_map and load normal_map image
						if (normal_map_path.is_relative())
							normal_map_path = path.parent_path() / normal_map_path;
						normal_map = mr_world.Container<World::ObjectType::NormalMap>().Create(
							ConStruct<NormalMap>(normal_map_path.stem().string(),
								LoadMap<World::ObjectType::NormalMap>(normal_map_path.string())));
						loaded_normal_maps.push_back(normal_map);
						material.SetNormalMap(normal_map);
					}
				}
				else if (parameter == "map_Pm")
				{
					// extract metalness map path from parameter
					std::string map_string;
					std::getline(ss, map_string);
					trim_spaces(map_string);
					std::filesystem::path metalness_map_path =
						extract_map_path(map_string);

					// search for already loaded metalness map with the same file name
					Handle<MetalnessMap> metalness_map;
					for (auto& mm : loaded_metalness_maps)
					{
						if (mm->GetName() == metalness_map_path.stem().string())
							metalness_map = mm;
					}

					if (metalness_map)
					{	// metalness map with the name has been loaded - share metalness map
						material.SetMetalnessMap(metalness_map);
					}
					else
					{	// metalness map hasn't been loaded yet - create new metalness map and load metalness map image
						if (metalness_map_path.is_relative())
							metalness_map_path = path.parent_path() / metalness_map_path;
						metalness_map = mr_world.Container<World::ObjectType::MetalnessMap>().Create(
							ConStruct<MetalnessMap>(metalness_map_path.stem().string(),
								LoadMap<World::ObjectType::MetalnessMap>(metalness_map_path.string())));
						loaded_metalness_maps.push_back(metalness_map);
						material.SetMetalnessMap(metalness_map);
					}
				}
				else if (parameter == "map_Pr")
				{
					// extract roughness map path from parameter
					std::string map_string;
					std::getline(ss, map_string);
					trim_spaces(map_string);
					std::filesystem::path roughness_map_path =
						extract_map_path(map_string);

					// search for already loaded roughness map with the same file name
					Handle<RoughnessMap> roughness_map;
					for (auto& rm : loaded_roughness_maps)
					{
						if (rm->GetName() == roughness_map_path.stem().string())
							roughness_map = rm;
					}

					if (roughness_map)
					{	// roughness map with the name has been loaded - share roughness map
						material.SetRoughnessMap(roughness_map);
					}
					else
					{	// roughness map hasn't been loaded yet - create new roughness map and load roughness map image
						if (roughness_map_path.is_relative())
							roughness_map_path = path.parent_path() / roughness_map_path;
						roughness_map = mr_world.Container<World::ObjectType::RoughnessMap>().Create(
							ConStruct<RoughnessMap>(roughness_map_path.stem().string(),
								LoadMap<World::ObjectType::RoughnessMap>(roughness_map_path.string())));
						loaded_roughness_maps.push_back(roughness_map);
						material.SetRoughnessMap(roughness_map);
					}
				}
			}
		}
	}
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~



	// ~~~~~~~~ OBJLoader ~~~~~~~~
	OBJLoader::OBJLoader(World& world)
		: MTLLoader(world)
	{}

	Handle<Group> OBJLoader::LoadOBJ(const std::filesystem::path& path)
	{
		if (path.extension().string() != ".obj")
			throw RayZath::Exception(
				"File path \"" + path.string() +
				"\" does not contain a valid .obj file.");

		// open specified file
		std::ifstream ifs(path, std::ios_base::in);
		if (!ifs.is_open())
			throw RayZath::Exception(
				"Failed to open file " + path.string());

		auto trim_spaces = [](std::string& s)
		{
			const size_t first = s.find_first_not_of(' ');
			if (first == std::string::npos) return;

			const size_t last = s.find_last_not_of(' ');
			s = s.substr(first, (last - first + 1));
		};

		std::vector<Vertex> vertices;
		std::vector<Texcrd> texcrds;
		std::vector<Normal> normals;
		uint32_t v_total = 0u, t_total = 0u, n_total = 0u;

		Handle<Material> material;
		uint32_t material_count = 0u;
		uint32_t material_idx = 0u;

		Handle<Mesh> object;
		Handle<Group> group = mr_world.Container<World::ObjectType::Group>().Create(ConStruct<Group>(path.stem().string()));
		if (!group) throw Exception("failed to create group for file: " + path.stem().string());

		std::string file_line;
		while (std::getline(ifs, file_line))
		{
			trim_spaces(file_line);
			if (file_line.empty()) continue;

			std::stringstream ss(file_line);
			std::string parameter;
			ss >> parameter;

			if (parameter == "mtllib")
			{
				std::string library_file_name;
				std::getline(ss, library_file_name);
				trim_spaces(library_file_name);
				std::filesystem::path library_file_path(library_file_name);
				if (!library_file_path.has_root_path())
					library_file_path = path.parent_path() / library_file_path;
				LoadMTL(library_file_path.string());
			}
			else if (parameter == "usemtl")
			{
				std::string name;
				std::getline(ss, name);
				if (name.size() > 0u)
					if (name[0] == ' ')
						name.erase(name.begin());

				if (object)
				{
					// check if material with given name is observed by current object
					uint32_t idx = object->GetMaterialIdx(name);
					if (idx >= object->GetMaterialCapacity())
					{
						// if object is refering to maximum number of materials
						// it can refer to
						if (material_count >= object->GetMaterialCapacity())
						{
							ThrowException("Object tried to refer to more than " +
								std::to_string(object->GetMaterialCapacity()) + " materials.");
						}

						// material with given name not found in current object,
						// search for material with this name in world container
						auto mat = mr_world.Container<World::ObjectType::Material>()[name];
						if (mat)
						{
							// material with given name has been found in world
							// container, add it to current object
							object->SetMaterial(mat, material_count);
							material_idx = material_count;
							material_count++;
						}
						else
						{
							// there is no material with given name in world container
							ThrowException("Object tried to use nonexistent/not yet loaded material.");
						}
					}
					else
					{
						// material with given name found in current object,
						// every subsequent face will refer to material
						// observed by object at material_idx index
						material_idx = idx;
					}
				}
			}
			else if (parameter == "o" || parameter == "g")
			{
				if (object)
				{
					// increase total elements readed
					v_total += object->GetStructure()->GetVertices().GetCount();
					t_total += object->GetStructure()->GetTexcrds().GetCount();
					n_total += object->GetStructure()->GetNormals().GetCount();

					// erase elements already stored in current object making them 
					// no longer available
					vertices.erase(vertices.begin(), vertices.begin() + object->GetStructure()->GetVertices().GetCount());
					texcrds.erase(texcrds.begin(), texcrds.begin() + object->GetStructure()->GetTexcrds().GetCount());
					normals.erase(normals.begin(), normals.begin() + object->GetStructure()->GetNormals().GetCount());
				}

				// create new object with empty MeshStructure
				ConStruct<Mesh> construct;
				std::getline(ss, construct.name);
				construct.mesh_structure =
					mr_world.Container<World::ObjectType::MeshStructure>().Create(ConStruct<MeshStructure>(construct.name));
				object = mr_world.Container<World::ObjectType::Mesh>().Create(construct);
				Group::link(group, object);
				material_count = 0u;
			}

			else if (parameter == "v")
			{
				Math::vec3f v;
				ss >> v.x >> v.y >> v.z;
				v.z = -v.z;
				vertices.push_back(v);
			}
			else if (parameter == "vt")
			{
				Math::vec2f t;
				ss >> t.x >> t.y;
				texcrds.push_back(t);
			}
			else if (parameter == "vn")
			{
				Math::vec3f n;
				ss >> n.x >> n.y >> n.z;
				n.z = -n.z;
				normals.push_back(n);
			}
			else if (parameter == "f")
			{
				constexpr uint32_t max_n_gon = 8u;

				// extract vertices data to separate strings
				std::string vertex_as_string[max_n_gon];
				uint8_t face_v_count = 0u;
				while (!ss.eof() && face_v_count < max_n_gon)
				{
					ss >> vertex_as_string[face_v_count];
					face_v_count++;
				}

				// allocate vertex data buffers
				uint32_t v[max_n_gon];
				uint32_t t[max_n_gon];
				uint32_t n[max_n_gon];
				for (uint32_t i = 0u; i < max_n_gon; i++)
				{
					v[i] = ComponentContainer<Vertex>::GetEndPos();
					t[i] = ComponentContainer<Texcrd>::GetEndPos();
					n[i] = ComponentContainer<Normal>::GetEndPos();
				}

				for (uint8_t vertex_idx = 0u; vertex_idx < face_v_count; vertex_idx++)
				{
					auto decompose_vertex_description = [](const std::string& vertex_description)
					{
						std::array<std::string, 3u> indices;
						std::string::const_iterator first = vertex_description.begin(), last;
						for (size_t i = 0u; i < 3u; i++)
						{
							last = std::find(first, vertex_description.end(), '/');
							indices[i] = vertex_description.substr(first - vertex_description.begin(), last - first);
							if (last == vertex_description.end())
								return indices;
							first = last + 1u;
						}
						return indices;
					};
					std::array<std::string, 3u> indices = decompose_vertex_description(vertex_as_string[vertex_idx]);

					// convert position index
					if (!indices[0].empty())
					{
						int32_t vp_idx = std::stoi(indices[0]);
						if (vp_idx > 0 && vp_idx <= int32_t(vertices.size() + v_total))
						{
							v[vertex_idx] = vp_idx - 1;
						}
						else if (vp_idx < 0 && int32_t(vertices.size() + v_total) + vp_idx >= 0)
						{
							v[vertex_idx] = int32_t(vertices.size() + v_total) + vp_idx;
						}
					}

					// convert texcrd index
					if (!indices[1].empty())
					{
						int32_t vt_idx = std::stoi(indices[1]);
						if (vt_idx > 0 && vt_idx <= int32_t(texcrds.size() + t_total))
						{
							t[vertex_idx] = vt_idx - 1;
						}
						else if (vt_idx < 0 && int32_t(texcrds.size() + t_total) + vt_idx >= 0)
						{
							t[vertex_idx] = int32_t(texcrds.size() + t_total) + vt_idx;
						}
					}

					// convert normal index
					if (!indices[2].empty())
					{
						int32_t vn_idx = std::stoi(indices[2]);
						if (vn_idx > 0 && vn_idx <= int32_t(normals.size() + n_total))
						{
							n[vertex_idx] = vn_idx - 1;
						}
						else if (vn_idx < 0 && int32_t(normals.size() + n_total) + vn_idx >= 0)
						{
							n[vertex_idx] = int32_t(normals.size() + n_total) + vn_idx;
						}
					}
				}

				// insert vertices, texcrds and normals to current object up to 
				// indices referenced in current polygon
				{
					// find how many components have to be inserted
					uint32_t v_max = 0u, t_max = 0u, n_max = 0u;
					for (uint32_t i = 0u; i < face_v_count; i++)
					{
						// check if components' indices are valid
						if (v[i] != ComponentContainer<Vertex>::GetEndPos() && v[i] >= v_total)
							v_max = std::max(v_max, (v[i] -= v_total) + 1u);
						if (t[i] != ComponentContainer<Texcrd>::GetEndPos() && t[i] >= t_total)
							t_max = std::max(t_max, (t[i] -= t_total) + 1u);
						if (n[i] != ComponentContainer<Normal>::GetEndPos() && n[i] >= n_total)
							n_max = std::max(n_max, (n[i] -= n_total) + 1u);
					}

					// insert into structure minimum ammount of components to
					// be able to insert current face
					for (uint32_t i = object->GetStructure()->GetVertices().GetCount();
						i < v_max;
						i++)
					{
						object->GetStructure()->CreateVertex(vertices[i]);
					}
					for (uint32_t i = object->GetStructure()->GetTexcrds().GetCount();
						i < t_max;
						i++)
					{
						object->GetStructure()->CreateTexcrd(texcrds[i]);
					}
					for (uint32_t i = object->GetStructure()->GetNormals().GetCount();
						i < n_max;
						i++)
					{
						object->GetStructure()->CreateNormal(normals[i]);
					}
				}

				// create face
				if (face_v_count == 3u)
				{	// triangle					

					object->GetStructure()->CreateTriangle(
						v[0], v[2], v[1],
						t[0], t[2], t[1],
						n[0], n[2], n[1],
						material_idx);
				}
				else if (face_v_count == 4u)
				{	// quadrilateral

					// for now just split quad into two touching triangles
					object->GetStructure()->CreateTriangle(
						v[0], v[2], v[1],
						t[0], t[2], t[1],
						n[0], n[2], n[1],
						material_idx);

					object->GetStructure()->CreateTriangle(
						v[0], v[3], v[2],
						t[0], t[3], t[2],
						n[0], n[3], n[2],
						material_idx);
				}
				else
				{	// polygon (tesselate into triangles)
					for (uint8_t i = 1u; i < face_v_count - 1u; i++)
					{
						object->GetStructure()->CreateTriangle(
							v[0], v[i + 1u], v[i],
							t[0], t[i + 1u], t[i],
							n[0], n[i + 1u], n[i],
							material_idx);
					}
				}
			}
		}

		return group;
	}
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~



	// ~~~~~~~~ Loader ~~~~~~~~
	Loader::Loader(World& world)
		: OBJLoader(world)
		, mp_json_loader(new JsonLoader(world))
	{}

	void Loader::LoadScene(const std::filesystem::path& path)
	{
		if (path.extension().string() != ".json")
			throw Exception(
				"File path \"" + path.string() +
				"\" does not contain a valid .json file.");

		// open specified file
		std::ifstream ifs(path, std::ios_base::in);
		if (!ifs.is_open())
			throw Exception(
				"Failed to open file " + path.string());

		mp_json_loader->LoadJsonScene(ifs, path);
	}
	// ~~~~~~~~~~~~~~~~~~~~~~~~
}