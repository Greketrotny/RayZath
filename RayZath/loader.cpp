#include "loader.h"

#include "./lib/include/CImg.h"

#include <fstream>
#include <sstream>

namespace RayZath
{
	// ~~~~~~~~ LoaderBase ~~~~~~~~
	LoaderBase::LoaderBase(World& world)
		: mr_world(world)
	{}

	std::array<std::string, 3ull> LoaderBase::ParseFileName(const std::string& file_name)
	{
		std::string extension, name, path;

		const size_t dot_idx = file_name.find_last_of('.');
		size_t last_slash_idx = 
			std::string(file_name.begin(), file_name.begin() + std::min(dot_idx, file_name.size())).
			find_last_of('/');

		if (dot_idx == std::string::npos) return {};	// no dot in the file_name (invalid name)
		extension = file_name.substr(dot_idx + 1ull, file_name.size() - dot_idx);

		if (last_slash_idx == std::string::npos)
		{
			name = file_name.substr(0ull, dot_idx);
		}
		else
		{
			last_slash_idx++;
			path = file_name.substr(0ull, last_slash_idx);
			name = file_name.substr(last_slash_idx, dot_idx - last_slash_idx);
		}

		return { path, name, extension };
	}
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~



	// ~~~~~~~~ BitmapLoader ~~~~~~~~
	BitmapLoader::BitmapLoader(World& world)
		: LoaderBase(world)
	{}

	Graphics::Bitmap BitmapLoader::LoadTexture(const std::string& path)
	{
		cimg_library::cimg::imagemagick_path(
			"D:/Program Files/ImageMagick-7.0.10-53-portable-Q8-x64/convert.exe");
		cil::CImg<unsigned char> image(path.c_str());

		Graphics::Bitmap texture(image.width(), image.height());
		if (image.spectrum() == 3)
		{
			for (int x = 0; x < texture.GetWidth(); x++)
			{
				for (int y = 0; y < texture.GetHeight(); y++)
				{
					texture.Value(x, y) =
						Graphics::Color(
							*image.data(x, y, 0, 0),
							*image.data(x, y, 0, 1),
							*image.data(x, y, 0, 2), 0xFF);
				}
			}
		}
		else if (image.spectrum() == 1)
		{
			for (int x = 0; x < texture.GetWidth(); x++)
			{
				for (int y = 0; y < texture.GetHeight(); y++)
				{
					auto& value = *image.data(x, y, 0, 0);
					texture.Value(x, y) =
						Graphics::Color(value);
				}
			}
		}

		return texture;
	}
	Graphics::Bitmap BitmapLoader::LoadNormalMap(const std::string& path)
	{
		return LoadTexture(path);
	}
	Graphics::Buffer2D<uint8_t> BitmapLoader::LoadSpecularityMap(const std::string& path)
	{
		cimg_library::cimg::imagemagick_path(
			"D:/Program Files/ImageMagick-7.0.10-53-portable-Q8-x64/convert.exe");
		cil::CImg<unsigned char> image(path.c_str());

		Graphics::Buffer2D<uint8_t> specularity_map(image.width(), image.height());
		if (image.spectrum() > 0)
		{
			for (int x = 0; x < specularity_map.GetWidth(); x++)
			{
				for (int y = 0; y < specularity_map.GetHeight(); y++)
				{
					specularity_map.Value(x, y) = *image.data(x, y, 0, 0);
				}
			}
		}

		return specularity_map;
	}
	Graphics::Buffer2D<uint8_t> BitmapLoader::LoadRoughnessMap(const std::string& path)
	{
		cimg_library::cimg::imagemagick_path(
			"D:/Program Files/ImageMagick-7.0.10-53-portable-Q8-x64/convert.exe");
		cil::CImg<unsigned char> image(path.c_str());

		Graphics::Buffer2D<uint8_t> roughness_map(image.width(), image.height());
		if (image.spectrum() > 0)
		{
			for (int x = 0; x < roughness_map.GetWidth(); x++)
			{
				for (int y = 0; y < roughness_map.GetHeight(); y++)
				{
					roughness_map.Value(x, y) = *image.data(x, y, 0, 0);
				}
			}
		}

		return roughness_map;
	}
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



	// ~~~~~~~~ MTLLoader ~~~~~~~~
	MTLLoader::MTLLoader(World& world)
		: BitmapLoader(world)
	{}

	std::vector<Handle<Material>> MTLLoader::LoadMTL(const std::string& full_file_name)
	{
		// decompose file extension
		auto [mtl_path, mtl_file_name, mtl_extension] = ParseFileName(full_file_name);
		if (mtl_extension != "mtl")
			throw RayZath::Exception(
				"File \"" + mtl_file_name + "." + mtl_extension +
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

		std::vector<Handle<Material>> loaded_materials;
		std::vector<Handle<Texture>> loaded_textures;
		std::vector<Handle<NormalMap>> loaded_normal_maps;
		std::vector<Handle<SpecularityMap>> loaded_specularity_maps;
		std::vector<Handle<RoughnessMap>> loaded_roughness_maps;
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
					material = mr_world.Container<World::ContainerType::Material>().Create(
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
					ss >> material_name;
					material = mr_world.Container<World::ContainerType::Material>().Create(
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
				else if (parameter == "Ks")
				{	// material specularity
					// for simplicity only the first value from three color channels 
					// is taken for consideration as specularity factor

					float specularity = 0.0f;
					ss >> specularity;
					material->SetSpecularity(specularity);
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
							auto& values_iter = std::next(curr_option);
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
					std::string texture_path =
						extract_map_path({ file_line.begin() + parameter.size(), file_line.end() });

					// decompose texture path
					auto [path, name, ext] = ParseFileName(texture_path);

					// search for already loaded texture with the same file name
					Handle<Texture> texture;
					for (auto& t : loaded_textures)
					{
						if (t->GetName() == name)
							texture = t;
					}

					if (texture)
					{	// texture with the name has been loaded - share texture
						material->SetTexture(texture);
					}
					else
					{	// texture hasn't been loaded yet - create new texture and load texture image
						texture = mr_world.Container<World::ContainerType::Texture>().Create(
							ConStruct<Texture>(name,
								LoadTexture(mtl_path + path + name + "." + ext)));
						loaded_textures.push_back(texture);
						material->SetTexture(texture);
					}
				}
				else if (parameter == "norm")
				{
					// extract normal map path from parameter
					std::string normal_map_path =
						extract_map_path({ file_line.begin() + parameter.size(), file_line.end() });

					// decompose normal map file path
					auto [path, name, ext] = ParseFileName(normal_map_path);

					// search for already loaded normal map with the same file name
					Handle<NormalMap> normal_map;
					for (auto& nm : loaded_normal_maps)
					{
						if (nm->GetName() == name)
							normal_map = nm;
					}

					if (normal_map)
					{	// normal_map with the name has been loaded - share normal_map
						material->SetNormalMap(normal_map);
					}
					else
					{	// normal_map hasn't been loaded yet - create new normal_map and load normal_map image
						normal_map = mr_world.Container<World::ContainerType::NormalMap>().Create(
							ConStruct<NormalMap>(name,
								LoadNormalMap(mtl_path + path + name + "." + ext)));
						loaded_normal_maps.push_back(normal_map);
						material->SetNormalMap(normal_map);
					}
				}
				else if (parameter == "map_Ks")
				{
					// extract specularity map path from parameter
					std::string specularity_map_path =
						extract_map_path({ file_line.begin() + parameter.size(), file_line.end() });

					// decompose specularity map file path
					auto [path, name, ext] = ParseFileName(specularity_map_path);

					// search for already loaded specularity map with the same file name
					Handle<SpecularityMap> specularity_map;
					for (auto& sm : loaded_specularity_maps)
					{
						if (sm->GetName() == name)
							specularity_map = sm;
					}

					if (specularity_map)
					{	// specularity map with the name has been loaded - share specularity map
						material->SetSpecularityMap(specularity_map);
					}
					else
					{	// specularity map hasn't been loaded yet - create new specularity map and load specularity map image
						specularity_map = mr_world.Container<World::ContainerType::SpecularityMap>().Create(
							ConStruct<SpecularityMap>(name,
								LoadSpecularityMap(mtl_path + path + name + "." + ext)));
						loaded_specularity_maps.push_back(specularity_map);
						material->SetSpecularityMap(specularity_map);
					}
				}
				else if (parameter == "map_Pr")
				{
					// extract roughness map path from parameter
					std::string roughness_map_path =
						extract_map_path({ file_line.begin() + parameter.size(), file_line.end() });

					// decompose roughness map file path
					auto [path, name, ext] = ParseFileName(roughness_map_path);

					// search for already loaded roughness map with the same file name
					Handle<RoughnessMap> roughness_map;
					for (auto& rm : loaded_roughness_maps)
					{
						if (rm->GetName() == name)
							roughness_map = rm;
					}

					if (roughness_map)
					{	// roughness map with the name has been loaded - share roughness map
						material->SetRoughnessMap(roughness_map);
					}
					else
					{	// roughness map hasn't been loaded yet - create new roughness map and load roughness map image
						roughness_map = mr_world.Container<World::ContainerType::RoughnessMap>().Create(
							ConStruct<RoughnessMap>(name,
								LoadRoughnessMap(mtl_path + path + name + "." + ext)));
						loaded_roughness_maps.push_back(roughness_map);
						material->SetRoughnessMap(roughness_map);
					}
				}
			}
		}

		ifs.close();

		return loaded_materials;
	}
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~



	// ~~~~~~~~ OBJLoader ~~~~~~~~
	OBJLoader::OBJLoader(World& world)
		: MTLLoader(world)
	{}
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~



	// ~~~~~~~~ Loader ~~~~~~~~
	Loader::Loader(World& world)
		: OBJLoader(world)
	{}
	// ~~~~~~~~~~~~~~~~~~~~~~~~
}