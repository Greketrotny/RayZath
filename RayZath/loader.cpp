#include "loader.hpp"

#include "json_loader.hpp"
#include "world.hpp"
#include "rzexception.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include "./lib/stb_image/stb_image.h"

#include <fstream>
#include <sstream>
#include <string>
#include <memory>
#include <algorithm>

namespace RayZath::Engine
{
	LoaderBase::LoaderBase(World& world)
		: mr_world(world)
	{}


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
		RZAssert(data, "failed to open " + path);
		RZAssert(width != 0 && height != 0, "one dimension had size of 0");

		Graphics::Bitmap image(width, height, {});
		std::memcpy((void*)image.GetMapAddress(), data.get(),
			image.GetWidth() * image.GetHeight() * sizeof(*image.GetMapAddress()));
		return image;
	}
	template <>
	Graphics::Bitmap BitmapLoader::LoadMap<World::ObjectType::NormalMap>(const std::string& path)
	{
		auto bitmap = LoadMap<World::ObjectType::Texture>(path);
		for (size_t i = 0; i < bitmap.GetHeight(); i++)
		{
			for (size_t j = 0; j < bitmap.GetWidth(); j++)
			{
				auto& value = bitmap.Value(j, i);
				value.green = -value.green;
			}
		}
		return bitmap;
	}
	template <>
	Graphics::Buffer2D<uint8_t> BitmapLoader::LoadMap<World::ObjectType::MetalnessMap>(const std::string& path)
	{
		int width{}, height{}, components{};

		std::unique_ptr<stbi_uc, decltype(&stbi_image_free)> data(
			stbi_load(path.c_str(), &width, &height, &components, 1),
			&stbi_image_free);
		RZAssert(data, "failed to open " + path);
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
		RZAssert(data, "failed to open " + path);
		RZAssert(width != 0 && height != 0, "one dimension had size of 0");

		Graphics::Buffer2D<float> image(width, height, {});
		std::memcpy((void*)image.GetMapAddress(), data.get(),
			image.GetWidth() * image.GetHeight() * sizeof(*image.GetMapAddress()));
		return image;
	}


	template <World::ObjectType T>
	struct TypeIdentity
	{
		static constexpr auto value = T;
	};

	MTLLoader::MTLLoader(World& world)
		: BitmapLoader(world)
	{}

	std::vector<Handle<Material>> MTLLoader::LoadMTL(const std::filesystem::path& file_path)
	{
		LoadedSet<
			World::ObjectType::Texture,
			World::ObjectType::NormalMap,
			World::ObjectType::MetalnessMap,
			World::ObjectType::RoughnessMap,
			World::ObjectType::EmissionMap> loaded_set;
		auto loaded_set_view = loaded_set.createView<
			World::ObjectType::Texture,
			World::ObjectType::NormalMap,
			World::ObjectType::MetalnessMap,
			World::ObjectType::RoughnessMap,
			World::ObjectType::EmissionMap>();
		LoadResult load_result;

		RZAssert(file_path.has_filename(), file_path.string() + " doesn't contain file name");
		RZAssert(file_path.has_extension() && file_path.extension() == ".mtl", file_path.string() + " doesn't have .mtl extension");

		auto mat_descs = parseMTL(file_path, load_result);
		if (mat_descs.empty())
		{
			load_result.logError("Parsing " + file_path.string() + " returned no material.");
			return {};
		}

		std::vector<Handle<Material>> materials;
		for (const auto& mat_desc : mat_descs)
		{
			try
			{
				auto load_map = [&](
					const auto identity,
					const MatDesc::MapDesc& desc)
				{
					using map_t = World::object_t<decltype(identity)::value>;
					auto map = loaded_set_view.fetch<decltype(identity)::value>(desc.path);
					if (!map)
					{
						const auto& load_path =
							desc.path.is_absolute() ?
							desc.path :
							file_path.parent_path() / desc.path;
						auto bitmap{mr_world.GetLoader().LoadMap<decltype(identity)::value>(load_path.string())};
						map = mr_world.Container<decltype(identity)::value>().Create(
							ConStruct<map_t>(
								desc.path.filename().string(), std::move(bitmap),
								map_t::FilterMode::Point,
								map_t::AddressMode::Wrap,
								desc.scale,
								{},
								desc.origin));
						loaded_set_view.add<decltype(identity)::value>(desc.path, map);
						return map;
					}
					using type_t = decltype(map);
					return type_t{};
				};

				auto texture = mat_desc.texture ?
					load_map(TypeIdentity<World::ObjectType::Texture>{}, * mat_desc.texture) : Handle<Texture>{};
				auto normal_map = mat_desc.normal_map ?
					load_map(TypeIdentity<World::ObjectType::NormalMap>{}, * mat_desc.normal_map) : Handle<NormalMap>{};
				auto metalness_map = mat_desc.metalness_map ?
					load_map(TypeIdentity<World::ObjectType::MetalnessMap>{}, * mat_desc.metalness_map) : Handle<MetalnessMap>{};
				auto roughness_map = mat_desc.roughness_map ?
					load_map(TypeIdentity<World::ObjectType::RoughnessMap>{}, * mat_desc.roughness_map) : Handle<RoughnessMap>{};
				auto emission_map = mat_desc.emission_map ?
					load_map(TypeIdentity<World::ObjectType::EmissionMap>{}, * mat_desc.emission_map) : Handle<EmissionMap>{};


				// create material with properties parsed from file
				auto material{mr_world.Container<World::ObjectType::Material>().Create(mat_desc.properties)};
				RZAssertCore(material, "Failed to create material");

				material->SetTexture(texture);
				material->SetNormalMap(normal_map);
				material->SetMetalnessMap(metalness_map);
				material->SetRoughnessMap(roughness_map);
				material->SetEmissionMap(emission_map);
			}
			catch (Exception& e)
			{
				using namespace std::string_literals;
				load_result.logError("Failed to load map because: "s + e.what());
			}
		}

		return materials;
	}
	void MTLLoader::LoadMTL(const std::filesystem::path& file_path, Material& material)
	{
		LoadedSet<
			World::ObjectType::Texture,
			World::ObjectType::NormalMap,
			World::ObjectType::MetalnessMap,
			World::ObjectType::RoughnessMap,
			World::ObjectType::EmissionMap> loaded_set;
		auto loaded_set_view = loaded_set.createView<
			World::ObjectType::Texture,
			World::ObjectType::NormalMap,
			World::ObjectType::MetalnessMap,
			World::ObjectType::RoughnessMap,
			World::ObjectType::EmissionMap>();
		LoadResult load_result;

		RZAssert(file_path.has_filename(), file_path.string() + " doesn't contain file name");
		RZAssert(file_path.has_extension() && file_path.extension() == ".mtl", file_path.string() + " doesn't have .mtl extension");

		auto mat_descs = parseMTL(file_path, load_result);
		if (mat_descs.empty())
		{
			load_result.logError("Parsing " + file_path.string() + " returned no material.");
			return;
		}
		auto& mat_desc = mat_descs[0];

		try
		{
			auto load_map = [&](
				const auto identity,
				const MatDesc::MapDesc& desc)
			{
				using map_t = World::object_t<decltype(identity)::value>;
				auto map = loaded_set_view.fetch<decltype(identity)::value>(desc.path);
				if (!map)
				{
					const auto& load_path =
						desc.path.is_absolute() ?
						desc.path :
						file_path.parent_path() / desc.path;
					auto bitmap{mr_world.GetLoader().LoadMap<decltype(identity)::value>(load_path.string())};
					map = mr_world.Container<decltype(identity)::value>().Create(
						ConStruct<map_t>(
							desc.path.filename().string(), std::move(bitmap),
							map_t::FilterMode::Point,
							map_t::AddressMode::Wrap,
							desc.scale,
							{},
							desc.origin));
					loaded_set_view.add<decltype(identity)::value>(desc.path, map);
					return map;
				}
				using type_t = decltype(map);
				return type_t{};
			};

			auto texture = mat_desc.texture ?
				load_map(TypeIdentity<World::ObjectType::Texture>{}, * mat_desc.texture) : Handle<Texture>{};
			auto normal_map = mat_desc.normal_map ?
				load_map(TypeIdentity<World::ObjectType::NormalMap>{}, * mat_desc.normal_map) : Handle<NormalMap>{};
			auto metalness_map = mat_desc.metalness_map ?
				load_map(TypeIdentity<World::ObjectType::MetalnessMap>{}, * mat_desc.metalness_map) : Handle<MetalnessMap>{};
			auto roughness_map = mat_desc.roughness_map ?
				load_map(TypeIdentity<World::ObjectType::RoughnessMap>{}, * mat_desc.roughness_map) : Handle<RoughnessMap>{};
			auto emission_map = mat_desc.emission_map ?
				load_map(TypeIdentity<World::ObjectType::EmissionMap>{}, * mat_desc.emission_map) : Handle<EmissionMap>{};

			material.SetColor(mat_desc.properties.color);
			material.SetMetalness(mat_desc.properties.metalness);
			material.SetRoughness(mat_desc.properties.roughness);
			material.SetEmission(mat_desc.properties.emission);
			material.SetIOR(mat_desc.properties.ior);
			material.SetScattering(mat_desc.properties.scattering);

			material.SetTexture(texture);
			material.SetNormalMap(normal_map);
			material.SetMetalnessMap(metalness_map);
			material.SetRoughnessMap(roughness_map);
			material.SetEmissionMap(emission_map);
		}
		catch (Exception& e)
		{
			using namespace std::string_literals;
			load_result.logError("Failed to load map because: "s + e.what());
		}
	}

	std::vector<MTLLoader::MatDesc> MTLLoader::parseMTL(
		const std::filesystem::path& path,
		LoadResult& load_result)
	{
		// open .mtl file
		std::ifstream mtl_file(path, std::ios_base::in);
		RZAssert(mtl_file.is_open(), "Failed to open file " + path.string());

		std::vector<MatDesc> materials;
		uint32_t line_number = 0;

		auto trim_spaces = [](const std::string& s) -> std::string_view
		{
			const auto begin = std::find_if(s.begin(), s.end(), [](const auto& ch) { return !std::isspace(ch); });
			const auto end = std::find_if(s.rbegin(), s.rend(), [](const auto& ch) { return !std::isspace(ch); });
			return std::string_view{begin, end.base()};
		};
		auto parse_map_statement = [&](std::stringstream& map_statement) -> MatDesc::MapDesc
		{
			std::string statement_string;
			std::getline(map_statement, statement_string);
			MatDesc::MapDesc map;
			std::stringstream param_stream(statement_string);
			if (statement_string.empty())
			{
				load_result.logError(
					path.string() + ':' + std::to_string(line_number) + ": " +
					"Map statement was empty (At least file name required).");
				return {};
			}

			// parse parameters
			while (param_stream.good())
			{
				std::string param;
				param_stream >> param;

				if (param == "-o")
				{
					Math::vec2f32 origin;
					if (!(param_stream >> origin.x >> origin.y))
					{
						load_result.logError(
							path.string() + ':' + std::to_string(line_number) + ": " +
							"Invalid values for \"-o\" parameter. At least two numeric values are required.");
						continue;
					}
					map.origin = origin;
				}
				else if (param == "-s")
				{
					Math::vec2f32 scale;
					if (!(param_stream >> scale.x >> scale.y))
					{
						load_result.logError(
							path.string() + ':' + std::to_string(line_number) + ": " +
							"Invalid values for \"-s\" parameter. At least two numeric values are required.");
						continue;
					}
					map.scale = scale;
				}
			}

			// parse file name
			// try to find quoted string, and treat as full file name with path
			auto begin = statement_string.begin();
			while (begin != statement_string.end())
			{
				if (*begin == '"' && !(begin != statement_string.begin() && *std::prev(begin) == '\\'))
					break;
				begin++;
			}
			if (begin != statement_string.end())
			{
				auto end = begin + 1;
				while (end != statement_string.end())
				{
					if (*end == '"' && !(end != statement_string.begin() && *std::prev(end) == '\\'))
						break;
					end++;
				}

				if (end != statement_string.end())
				{
					// return path between non-escaped quotes
					map.path = std::filesystem::path(begin + 1, end);
					return map;
				}
			}

			// create path from the last token in map statement
			std::stringstream path_stream(statement_string);
			std::string last_token;
			while (path_stream.good())
			{
				path_stream >> last_token;
			}
			map.path = last_token;
			return map;
		};

		for (std::string file_line; std::getline(mtl_file, file_line); line_number++)
		{
			const auto line = trim_spaces(file_line);
			if (line.empty()) continue;

			std::stringstream line_stream{std::string(line)};

			std::string statement;
			line_stream >> statement;

			if (statement == "newmtl")
			{
				std::string material_name;
				std::getline(line_stream, material_name);
				material_name = trim_spaces(material_name);
				MatDesc desc{};
				desc.properties.name = std::move(material_name);
				materials.push_back(std::move(desc));
				continue;
			}
			if (materials.empty())
			{
				load_result.logWarning("First statement in file wasn't the \"newmtl\". Ignored.");
				continue;
			}
			auto& material = materials.back();

			if (statement == "Kd") // color
			{
				float values[3]{};
				if (!(line_stream >> values[0]))
				{
					load_result.logError(
						path.string() + ':' + std::to_string(line_number) + ": " +
						"invalid color specification (one or three numeric values [0.0, 1.0] required)");
					continue;
				}
				if (!(line_stream >> values[1]))
				{
					// Green and Blue the same as Red
					values[1] = values[2] = values[0];
				}
				else if (!(line_stream >> values[2]))
				{
					load_result.logError(
						path.string() + ':' + std::to_string(line_number) + ": " +
						"invalid blue value (one or three color numeric values [0.0, 1.0] required)");
					continue;
				}

				for (auto& value : values)
					value = std::clamp(value, 0.0f, 1.0f);

				material.properties.color.red = uint8_t(values[0] * 255.0f);
				material.properties.color.green = uint8_t(values[1] * 255.0f);
				material.properties.color.blue = uint8_t(values[2] * 255.0f);
			}
			else if (statement == "Ns") // roughness
			{
				float exponent = 0.0f;
				if (!(line_stream >> exponent))
				{
					load_result.logError(
						path.string() + ':' + std::to_string(line_number) + ": " +
						"Invalid exponent for \"Ns\" statement. Numeric value [1.0, 1000.0] required.");
					continue;
				}
				static constexpr auto max_exponent = 1000.0f;

				const float clamped = std::clamp(exponent, 1.0f, 1000.0f);
				if (clamped != exponent)
					load_result.logWarning(
						path.string() + ':' + std::to_string(line_number) + ": " +
						"Value " + std::to_string(clamped) + " is outside of [1.0, 1000.0] range. Clamped.");
				const float roughness = 1.0f - (log10f(clamped) / log10f(max_exponent));
				material.properties.roughness = roughness;
			}
			else if (statement == "d") // dissolve/opaque ( 1.0 - transparency)
			{
				float dissolve = 1.0f;
				if (!(line_stream >> dissolve))
				{
					load_result.logError(
						path.string() + ':' + std::to_string(line_number) + ": " +
						"Invalid paremeter for \"d\" statement. Numeric value in [0.0, 1.0] required.");
					continue;
				}

				const float clamped = std::clamp(dissolve, 0.0f, 1.0f);
				if (clamped != dissolve)
					load_result.logWarning(
						path.string() + ':' + std::to_string(line_number) + ": " +
						"Value " + std::to_string(clamped) + " is outside of [0.0, 1.0] range. Clamped.");

				material.properties.color.alpha = uint8_t(clamped * 255.0f);
			}
			else if (statement == "Tr") // transparency
			{
				float tr = 0.0f;
				if (!(line_stream >> tr))
				{
					load_result.logError(
						path.string() + ':' + std::to_string(line_number) + ": " +
						"Invalid paremeter for \"Tr\" statement. Numeric value in [0.0, 1.0] required.");
					continue;
				}

				const float clamped = std::clamp(tr, 0.0f, 1.0f);
				if (clamped != tr)
					load_result.logWarning(
						path.string() + ':' + std::to_string(line_number) + ": " +
						"Value " + std::to_string(clamped) + " is outside of [0.0, 1.0] range. Clamped.");

				material.properties.color.alpha = uint8_t((1.0f - clamped) * 255.0f);
			}
			else if (statement == "Ni") // IOR
			{
				float ior = 1.0f;
				if (!(line_stream >> ior))
				{
					load_result.logError(
						path.string() + ':' + std::to_string(line_number) + ": " +
						"Invalid paremeter for \"Ni\" statement. Numeric value >= 1.0 required.");
					continue;
				}

				const float clamped = ior < 1.0f ? 1.0f : ior;
				if (clamped != ior)
					load_result.logWarning(
						path.string() + ':' + std::to_string(line_number) + ": " +
						"Value " + std::to_string(clamped) + " for \"Ni\" was less than 1.0. Clamped.");
				material.properties.ior = clamped;
			}
			else if (statement == "Pm" || statement == "Pr") // metalness || roughness
			{
				float metalness = 0.0f;
				if (!(line_stream >> metalness))
				{
					load_result.logError(
						path.string() + ':' + std::to_string(line_number) + ": " +
						"Invalid paremeter for \"" + statement + "\" statement. Numeric value in[0.0, 1.0] required.");
					continue;
				}

				const float clamped = std::clamp(metalness, 0.0f, 1.0f);
				if (clamped != metalness)
					load_result.logWarning(
						path.string() + ':' + std::to_string(line_number) + ": " +
						"Value " + std::to_string(clamped) + " for \"" + statement + "\"  is outside of [0.0, 1.0] range. Clamped.");
				if (statement == "Pm")
					material.properties.metalness = clamped;
				else
					material.properties.roughness = clamped;
			}
			else if (statement == "Ke") // emission
			{
				float emission = 0.0f;
				if (!(line_stream >> emission))
				{
					load_result.logError(
						path.string() + ':' + std::to_string(line_number) + ": " +
						"Invalid paremeter for \"" + statement + "\" statement. Positive numeric value required.");
					continue;
				}

				const float clamped = emission < 0.0f ? 0.0f : emission;
				if (clamped != emission)
					load_result.logWarning(
						path.string() + ':' + std::to_string(line_number) + ": " +
						"Value " + std::to_string(clamped) + " for \"" + statement + "\"  is less than 0.0. Clamped.");
				material.properties.emission = clamped;
			}
			// maps
			else if (statement == "map_Kd") // texture
			{
				material.texture = parse_map_statement(line_stream);
			}
			else if (statement == "norm") // normal map
			{
				material.normal_map = parse_map_statement(line_stream);
			}
			else if (statement == "map_Pm") // metalness
			{
				material.metalness_map = parse_map_statement(line_stream);
			}
			else if (statement == "map_Pr") // roughness
			{
				material.roughness_map = parse_map_statement(line_stream);
			}
			else if (statement == "map_Ke") // texture
			{
				material.emission_map = parse_map_statement(line_stream);
			}
		}

		return materials;
	}


	// ~~~~~~~~ OBJLoader ~~~~~~~~
	OBJLoader::OBJLoader(World& world)
		: MTLLoader(world)
	{}

	Handle<Group> OBJLoader::LoadOBJ(const std::filesystem::path& path)
	{
		RZAssert(path.extension().string() == ".obj",
			"File path \"" + path.string() +
			"\" does not contain a valid .obj file.");

		// open specified file
		std::ifstream ifs(path, std::ios_base::in);
		RZAssert(ifs.is_open(), "Failed to open file " + path.string());

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
						RZAssert(material_count < object->GetMaterialCapacity(),
							"Object tried to refer to more than " +
							std::to_string(object->GetMaterialCapacity()) + " materials.");

						// material with given name not found in current object,
						// search for material with this name in world container
						auto mat = mr_world.Container<World::ObjectType::Material>()[name];
						RZAssertCore(mat, "Object tried to use nonexistent/not yet loaded material.");

						// material with given name has been found in world
						// container, add it to current object
						object->SetMaterial(mat, material_count);
						material_idx = material_count;
						material_count++;
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
		RZAssert(path.extension().string() == ".json",
			"File path \"" + path.string() +
			"\" does not contain a valid .json file.");

		// open specified file
		std::ifstream ifs(path, std::ios_base::in);
		RZAssert(ifs.is_open(), "Failed to open file " + path.string());

		mp_json_loader->LoadJsonScene(ifs, path);
	}
	// ~~~~~~~~~~~~~~~~~~~~~~~~
}