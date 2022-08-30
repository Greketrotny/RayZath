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

#include <iostream>

namespace RayZath::Engine
{
	LoaderBase::LoaderBase(World& world)
		: mr_world(world)
	{}

	std::string_view LoaderBase::trimSpaces(const std::string& s)
	{
		const auto begin = std::find_if(s.begin(), s.end(), [](const auto& ch) { return !std::isspace(ch); });
		const auto end = std::find_if(s.rbegin(), s.rend(), [](const auto& ch) { return !std::isspace(ch); });
		return std::string_view{begin, end.base()};
	}


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

		return LoadMTL(file_path, loaded_set_view, load_result);
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
	std::vector<Handle<Material>> MTLLoader::LoadMTL(
		const std::filesystem::path& file_path,
		MTLLoader::loaded_set_view_t loaded_set_view,
		LoadResult& load_result)
	{
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

				materials.push_back(std::move(material));
			}
			catch (Exception& e)
			{
				using namespace std::string_literals;
				load_result.logError("Failed to load map because: "s + e.what());
			}
		}

		return materials;
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

		std::set<std::string> unrecognized_statements{};
		for (std::string file_line; std::getline(mtl_file, file_line); line_number++)
		{
			const auto line = trimSpaces(file_line);
			if (line.empty()) continue;

			std::stringstream line_stream{std::string(line)};
			std::string statement;
			line_stream >> statement;

			if (statement == "#")
			{
				continue;
			}
			if (statement == "newmtl")
			{
				std::string material_name;
				std::getline(line_stream, material_name);
				material_name = trimSpaces(material_name);
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
			else
			{
				if (!unrecognized_statements.contains(statement))
				{
					load_result.logWarning("Unrecognized statement \"" + statement + "\".");
					unrecognized_statements.insert(std::move(statement));
				}
			}
		}

		return materials;
	}


	OBJLoader::OBJLoader(World& world)
		: MTLLoader(world)
	{}

	std::vector<Handle<MeshStructure>> OBJLoader::loadMeshes(const std::filesystem::path& file_path)
	{
		RZAssert(file_path.has_filename() && file_path.has_extension() && file_path.extension().string() == ".obj",
			"Path \"" + file_path.string() +
			"\" is not a valid path to .obj file.");

		LoadResult load_result;
		auto parse_result = parseOBJ(file_path, load_result);
		std::cout << load_result << std::endl;

		std::vector<Handle<MeshStructure>> meshes;
		for (auto& [mesh, material_ids] : parse_result.meshes)
			meshes.push_back(std::move(mesh));
		return meshes;
	}
	std::vector<Handle<Mesh>> OBJLoader::loadInstances(const std::filesystem::path& file_path)
	{
		RZAssert(file_path.has_filename() && file_path.has_extension() && file_path.extension().string() == ".obj",
			"Path \"" + file_path.string() +
			"\" is not a valid path to .obj file.");

		LoadResult load_result;
		auto parse_result = parseOBJ(file_path, load_result);

		LoadedSet<
			World::ObjectType::Texture,
			World::ObjectType::NormalMap,
			World::ObjectType::MetalnessMap,
			World::ObjectType::RoughnessMap,
			World::ObjectType::EmissionMap> loaded_set;

		// load material libraries
		std::map<std::string, Handle<Material>> materials;
		for (const auto& mtllib_path : parse_result.mtllibs)
		{
			try
			{
				const auto& path = mtllib_path.is_absolute() ? mtllib_path : file_path.parent_path() / mtllib_path;
				auto loaded_materials = LoadMTL(path, loaded_set.createView<
					World::ObjectType::Texture,
					World::ObjectType::NormalMap,
					World::ObjectType::MetalnessMap,
					World::ObjectType::RoughnessMap,
					World::ObjectType::EmissionMap>(), load_result);
				for (auto& loaded_material : loaded_materials)
				{
					auto [it, inserted] = materials.insert({loaded_material->GetName(), loaded_material});
					if (!inserted)
						load_result.logError(
							"The file \"" + file_path.string() +
							"\" declared usage of material libraries, which resulted in material name duplication (" +
							loaded_material->GetName() + ").");
				}
			}
			catch (Exception& e)
			{
				load_result.logError(e.what());
			}
		}

		// create instances
		std::vector<Handle<Mesh>> instances;
		for (const auto& [mesh, mesh_materials] : parse_result.meshes)
		{
			ConStruct<Mesh> desc(mesh->GetName());
			desc.mesh_structure = mesh;
			for (const auto& [material_name, material_idx] : mesh_materials)
			{
				const auto& material = materials.find(material_name);
				if (material == materials.end())
					load_result.logError("Failed to obtain \"" + material_name + "\" material.");
				else
					desc.material[material_idx] = material->second;
			}

			instances.push_back(mr_world.Container<World::ObjectType::Mesh>().Create(desc));
		}

		std::cout << load_result << std::endl;

		return instances;
	}
	Handle<Group> OBJLoader::LoadModel(const std::filesystem::path& file_path)
	{
		auto instances = loadInstances(file_path);

		// create enclosing group
		auto group{mr_world.Container<World::ObjectType::Group>().Create(ConStruct<Group>(file_path.filename().string()))};
		for (const auto& instance : instances)
			Group::link(group, instance);

		return group;
	}
	OBJLoader::ParseResult OBJLoader::parseOBJ(
		const std::filesystem::path& path,
		LoadResult& load_result)
	{
		// open specified file
		std::ifstream ifs(path, std::ios_base::in);
		RZAssert(ifs.is_open(), "Failed to open file " + path.string());

		ParseResult result;
		uint32_t material_count = 0, material_idx = 0;
		std::vector<Vertex> vertices;
		std::vector<Texcrd> texcrds;
		std::vector<Normal> normals;
		Math::vec2u32 vertex_range, texcrd_range, normal_range;

		auto shift_triangle_indices = [&](Handle<MeshStructure>& mesh)
		{
			for (uint32_t i = vertex_range.x; i < vertex_range.y; i++)
				mesh->CreateVertex(vertices[i]);
			for (uint32_t i = texcrd_range.x; i < texcrd_range.y; i++)
				mesh->CreateTexcrd(texcrds[i]);
			for (uint32_t i = normal_range.x; i < normal_range.y; i++)
				mesh->CreateNormal(normals[i]);

			for (uint32_t i = 0; i < mesh->GetTriangles().GetCount(); i++)
			{
				for (uint32_t c = 0; c < 3; c++)
				{
					mesh->GetTriangles()[i].vertices[c] -= vertex_range.x;
					mesh->GetTriangles()[i].texcrds[c] -= texcrd_range.x;
					mesh->GetTriangles()[i].normals[c] -= normal_range.x;
				}
			}
		};


		uint32_t line_number = 0;
		std::set<std::string> unrecognized_statements{};
		for (std::string file_line; std::getline(ifs, file_line); line_number++)
		{
			const auto line = trimSpaces(file_line);
			if (line.empty()) continue;

			std::stringstream line_stream{std::string(line)};
			std::string statement;
			line_stream >> statement;

			if (statement == "#")
			{
				continue;
			}
			else if (statement == "mtllib")
			{
				std::string library_file_name;
				std::getline(line_stream, library_file_name);
				result.mtllibs.insert(std::filesystem::path(trimSpaces(std::move(library_file_name))));
				continue;
			}
			else if (statement == "v")
			{
				Vertex v;
				if (!(line_stream >> v.x >> v.y >> v.z))
					load_result.logError("Vertex definition on line " + std::to_string(line_number) +
						" is invalid. Three numeric values are required.");
				v.z = -v.z; // .mtl right-handed to left-handed
				vertices.push_back(std::move(v));
				continue;
			}
			else if (statement == "vt")
			{
				Texcrd t;
				if (!(line_stream >> t.x >> t.y))
					load_result.logError("Texture coordinate definition on line " + std::to_string(line_number) +
						" is invalid. Two numeric values are required.");
				texcrds.push_back(std::move(t));
				continue;
			}
			else if (statement == "vn")
			{
				Normal n;
				if (!(line_stream >> n.x >> n.y >> n.z))
					load_result.logError("Vertex normal definition on line " + std::to_string(line_number) +
						" is invalid. Three numeric values are required.");
				n.z = -n.z; // .mtl right-handed to left-handed
				normals.push_back(std::move(n));
				continue;
			}
			else if (statement == "o" || statement == "g")
			{
				if (!result.meshes.empty())
				{
					auto& last_mesh = result.meshes.back().mesh;
					shift_triangle_indices(last_mesh);
				}

				std::string mesh_name;
				std::getline(line_stream, mesh_name);
				mesh_name = trimSpaces(mesh_name);

				auto mesh{mr_world.Container<World::ObjectType::MeshStructure>()
					.Create(ConStruct<MeshStructure>(std::move(mesh_name)))};
				result.meshes.push_back({std::move(mesh), {}});

				material_count = 0;
				material_idx = 0;
				vertex_range = texcrd_range = normal_range = Math::vec2u32(0, 0);
				continue;
			}

			if (result.meshes.empty())
			{
				load_result.logWarning(
					"Statement in line " + std::to_string(line_number) +
					" has to be preceded by object or group declaration. Ignored.");
				continue;
			}
			auto& [mesh, material_ids] = result.meshes.back();

			if (statement == "usemtl")
			{
				std::string material_name;
				std::getline(line_stream, material_name);
				material_name = trimSpaces(material_name);

				if (auto it = material_ids.find(material_name); it == material_ids.end())
				{
					if (material_count == Mesh::GetMaterialCapacity())
					{
						load_result.logWarning(
							"The declaration of usage of material \"" + material_name + "\" on line " +
							std::to_string(line_number) +
							" reached the limit of " + std::to_string(Mesh::GetMaterialCapacity()) +
							" materials per object. Ignored.");
					}
					else
					{
						material_idx = material_count;
						material_ids.insert(std::make_pair(std::move(material_name), material_count++));
					}
				}
				else
				{
					material_idx = it->second;
				}
			}
			else if (statement == "f")
			{
				using vertex_buff_t = std::string;
				using ids_strs_t = std::array<std::string_view, 3>;
				auto decompose_vertex_buff = [](const vertex_buff_t& buff)
				{
					ids_strs_t indices{};
					auto begin = buff.begin();
					for (size_t i = 0; i < indices.size(); i++)
					{
						auto end = std::find(begin, buff.end(), '/');
						indices[i] = std::string_view(begin, end);
						if (end == buff.end())
							break;
						begin = end + 1;
					}
					return indices;
				};

				// parse vertex indices
				constexpr size_t max_n_gon = 8;
				constexpr auto idx_unused = 0;
				vertex_buff_t buff{};
				uint32_t face_v_count = 0;
				std::array<std::array<int32_t, 3>, max_n_gon> indices{};
				for (; line_stream >> buff && face_v_count < max_n_gon; face_v_count++)
				{
					auto ids_strs = decompose_vertex_buff(buff);
					for (size_t i = 0; i < ids_strs.size(); i++)
					{
						const auto& idx_str = ids_strs[i];
						if (idx_str.empty()) continue;
						auto conv_result = std::from_chars(idx_str.data(), idx_str.data() + idx_str.size(), indices[face_v_count][i]);
						if (conv_result.ec != std::errc{})
						{
							load_result.logError(
								"Definition of face on line " + std::to_string(line_number) +
								": one of defined indices of " + std::to_string(face_v_count) + " vertex is invalid.");
							indices[i][face_v_count] = idx_unused;
						}
					}
				}
				if (face_v_count < 3)
				{
					load_result.logError(
						"On line " + std::to_string(line_number) +
						": at least three vertex indices description are required to create a valid face.");
					continue;
				}

				// check if indices are in valid range and negate negative ones. Translate to index triplets
				std::array<MeshStructure::triple_index_t, max_n_gon> index_triplets;
				for (uint8_t vertex_ids_idx = 0u; vertex_ids_idx < face_v_count; vertex_ids_idx++)
				{
					const auto& vertex_ids = indices[vertex_ids_idx];

					const auto& vertex_idx = vertex_ids[0];
					if (vertex_idx > 0 && uint32_t(vertex_idx) <= vertices.size())
						index_triplets[vertex_ids_idx][0] = vertex_idx - 1;
					else if (vertex_idx < 0 && uint32_t(-vertex_idx) <= vertices.size())
						index_triplets[vertex_ids_idx][0] = uint32_t(vertices.size()) - uint32_t(-vertex_idx);
					else
					{
						index_triplets[vertex_ids_idx][0] = ComponentContainer<Vertex>::sm_npos;
						if (vertex_idx != 0) load_result.logError(
							"On line " + std::to_string(line_number) + ": vertex index outside of range.");
					}
					const auto& texcrd_idx = vertex_ids[1];
					if (texcrd_idx > 0 && uint32_t(texcrd_idx) <= texcrds.size())
						index_triplets[vertex_ids_idx][1] = texcrd_idx - 1;
					else if (texcrd_idx < 0 && uint32_t(-texcrd_idx) <= texcrds.size())
						index_triplets[vertex_ids_idx][1] = uint32_t(texcrds.size()) - uint32_t(-texcrd_idx);
					else
					{
						index_triplets[vertex_ids_idx][1] = ComponentContainer<Texcrd>::sm_npos;
						if (texcrd_idx != 0) load_result.logError(
							"On line " + std::to_string(line_number) + ": texture coordinate index outside of range.");
					}
					auto& normal_idx = vertex_ids[2];
					if (normal_idx > 0 && uint32_t(normal_idx) <= normals.size())
						index_triplets[vertex_ids_idx][2] = normal_idx - 1;
					else if (normal_idx < 0 && uint32_t(-normal_idx) <= normals.size())
						index_triplets[vertex_ids_idx][2] = uint32_t(normals.size()) - uint32_t(-normal_idx);
					else
					{
						index_triplets[vertex_ids_idx][2] = ComponentContainer<Normal>::sm_npos;
						if (normal_idx != 0) load_result.logError(
							"On line " + std::to_string(line_number) + ": normal index outside of range.");
					}
				}

				// update component ranges
				for (uint32_t vertex_idx = 0; vertex_idx < face_v_count; vertex_idx++)
				{
					const auto& [v_idx, t_idx, n_idx] = index_triplets[vertex_idx];
					if (v_idx != ComponentContainer<Vertex>::sm_npos)
					{
						vertex_range.x = std::min(vertex_range.x, v_idx);
						vertex_range.y = std::max(vertex_range.y, v_idx + 1);
					}
					if (t_idx != ComponentContainer<Texcrd>::sm_npos)
					{
						texcrd_range.x = std::min(texcrd_range.x, t_idx);
						texcrd_range.y = std::max(texcrd_range.y, t_idx + 1);
					}
					if (n_idx != ComponentContainer<Normal>::sm_npos)
					{
						normal_range.x = std::min(normal_range.x, n_idx);
						normal_range.y = std::max(normal_range.y, n_idx + 1);
					}
				}

				// create triangles
				for (size_t i = 0; i < size_t(face_v_count - 2); i++)
				{
					mesh->CreateTriangle(
						index_triplets[0][0], index_triplets[i + 2u][0], index_triplets[i + 1u][0],
						index_triplets[0][1], index_triplets[i + 2u][1], index_triplets[i + 1u][1],
						index_triplets[0][2], index_triplets[i + 2u][2], index_triplets[i + 1u][2],
						material_idx);
				}
			}
			else
			{
				if (!unrecognized_statements.contains(statement))
				{
					load_result.logWarning("Unrecognized statement \"" + statement + "\".");
					unrecognized_statements.insert(std::move(statement));
				}
			}
		}

		if (!result.meshes.empty())
		{
			auto& last_mesh = result.meshes.back().mesh;
			shift_triangle_indices(last_mesh);
		}

		return result;
	}


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
}