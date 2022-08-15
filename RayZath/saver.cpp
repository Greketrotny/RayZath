#include "saver.hpp"
#include "json_saver.hpp"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "./lib/stb_image/stb_image_write.h"

#include <fstream>

namespace RayZath::Engine
{
	using namespace std::string_literals;

	std::filesystem::path SaverBase::relative_path(
		const std::filesystem::path& path,
		const std::filesystem::path& dest)
	{
		const auto& base_path = path.has_filename() ? path.parent_path() : path;
		return std::filesystem::relative(dest, base_path);
	}

	template<>
	std::filesystem::path BitmapSaver::SaveMap<World::ObjectType::Texture>(
		const Graphics::Bitmap& map,
		const std::filesystem::path& path,
		const std::string& file_name)
	{
		if (!std::filesystem::exists(path))
			std::filesystem::create_directories(path);
		auto full_path = path / (file_name + ".png");
		RZAssert(0 != stbi_write_png(
			full_path.string().c_str(),
			int(map.GetWidth()), int(map.GetHeight()),
			4,
			map.GetMapAddress(),
			int(map.GetWidth() * sizeof(*map.GetMapAddress()))),
			"failed to write image file to "s + full_path.string());
		return full_path;
	}
	template <>
	std::filesystem::path BitmapSaver::SaveMap<World::ObjectType::NormalMap>(
		const Graphics::Bitmap& map,
		const std::filesystem::path& path,
		const std::string& file_name)
	{
		return SaveMap<World::ObjectType::Texture>(map, path, file_name);
	}
	template <>
	std::filesystem::path BitmapSaver::SaveMap<World::ObjectType::MetalnessMap>(
		const Graphics::Buffer2D<uint8_t>& map,
		const std::filesystem::path& path,
		const std::string& file_name)
	{
		if (!std::filesystem::exists(path))
			std::filesystem::create_directories(path);
		std::filesystem::path full_path = path / (file_name + ".jpg");
		RZAssert(0 != stbi_write_png(
			full_path.string().c_str(),
			int(map.GetWidth()), int(map.GetHeight()),
			1,
			map.GetMapAddress(),
			int(map.GetWidth() * sizeof(*map.GetMapAddress()))),
			"failed to write image file to "s + full_path.string());
		return full_path;
	}
	template <>
	std::filesystem::path BitmapSaver::SaveMap<World::ObjectType::RoughnessMap>(
		const Graphics::Buffer2D<uint8_t>& map,
		const std::filesystem::path& path,
		const std::string& file_name)
	{
		if (!std::filesystem::exists(path))
			std::filesystem::create_directories(path);
		return SaveMap<World::ObjectType::MetalnessMap>(map, path, file_name);
	}
	template <>
	std::filesystem::path BitmapSaver::SaveMap<World::ObjectType::EmissionMap>(
		const Graphics::Buffer2D<float>& map,
		const std::filesystem::path& path,
		const std::string& file_name)
	{
		if (!std::filesystem::exists(path))
			std::filesystem::create_directories(path);
		std::filesystem::path full_path = path / (file_name + ".jpg");
		RZAssert(0 != stbi_write_hdr(
			full_path.string().c_str(),
			int(map.GetWidth()), int(map.GetHeight()),
			1,
			map.GetMapAddress()),
			"failed to write image file to "s + full_path.string());
		return full_path;
	}


	std::filesystem::path MTLSaver::SaveMTL(
		const Material& material,
		const std::filesystem::path& path,
		const std::string& file_name,
		const MapsPaths& maps_paths)
	{
		if (!std::filesystem::exists(path))
			std::filesystem::create_directories(path);
		const auto full_path = path / (file_name + ".mtl");
		try
		{
			std::ofstream file(full_path);
			RZAssert(file.is_open(), "failed to save material " + material.GetName() + " to " + full_path.string());
			file.exceptions(file.failbit);

			SaveMaterial(material, file, file_name, maps_paths);
		}
		catch (std::system_error&)
		{
			std::filesystem::remove(full_path);
			throw;
		}
		return full_path;
	}
	std::filesystem::path MTLSaver::SaveMTL(
		const std::vector<std::pair<std::reference_wrapper<Material>, std::string>>& materials,
		const std::filesystem::path& dir_path,
		const std::string& file_name)
	{
		RZAssert(!dir_path.has_filename(), "path can not contain file name");
		if (!std::filesystem::exists(dir_path))
			std::filesystem::create_directories(dir_path);
		const auto mtllib_path = dir_path / (file_name + ".mtl");

		ObjectNames<
			World::ObjectType::Texture,
			World::ObjectType::NormalMap,
			World::ObjectType::MetalnessMap,
			World::ObjectType::RoughnessMap,
			World::ObjectType::EmissionMap> map_names;

		// save all maps referenced by all materials without duplicates
		for (const auto& [material_ref, material_name] : materials)
		{
			const auto& material = material_ref.get();

			if (material.GetTexture() && !map_names.contains<World::ObjectType::Texture>(material.GetTexture()))
			{
				auto name = map_names.uniqueName<World::ObjectType::Texture>(
					material.GetTexture()->GetName() + "_color");
				auto path = SaveMap<World::ObjectType::Texture>(
					material.GetTexture()->GetBitmap(), dir_path, name);
				map_names.add<World::ObjectType::Texture>(material.GetTexture(), std::move(name), std::move(path));
			}
			if (material.GetNormalMap() && !map_names.contains<World::ObjectType::NormalMap>(material.GetNormalMap()))
			{
				auto name = map_names.uniqueName<World::ObjectType::NormalMap>(
					material.GetNormalMap()->GetName() + "_normal");
				auto path = SaveMap<World::ObjectType::NormalMap>(
					material.GetNormalMap()->GetBitmap(), dir_path, name);
				map_names.add<World::ObjectType::NormalMap>(material.GetNormalMap(), std::move(name), std::move(path));
			}
			if (material.GetMetalnessMap() && !map_names.contains<World::ObjectType::MetalnessMap>(material.GetMetalnessMap()))
			{
				auto name = map_names.uniqueName<World::ObjectType::MetalnessMap>(
					material.GetMetalnessMap()->GetName() + "_metalness");
				auto path = SaveMap<World::ObjectType::MetalnessMap>(
					material.GetMetalnessMap()->GetBitmap(), dir_path, name);
				map_names.add<World::ObjectType::MetalnessMap>(material.GetMetalnessMap(), std::move(name), std::move(path));
			}
			if (material.GetRoughnessMap() && !map_names.contains<World::ObjectType::RoughnessMap>(material.GetRoughnessMap()))
			{
				auto name = map_names.uniqueName<World::ObjectType::RoughnessMap>(
					material.GetRoughnessMap()->GetName() + "_roughness");
				auto path = SaveMap<World::ObjectType::RoughnessMap>(
					material.GetRoughnessMap()->GetBitmap(), dir_path, name);
				map_names.add<World::ObjectType::RoughnessMap>(material.GetRoughnessMap(), std::move(name), std::move(path));
			}
			if (material.GetEmissionMap() && !map_names.contains<World::ObjectType::EmissionMap>(material.GetEmissionMap()))
			{
				auto name = map_names.uniqueName<World::ObjectType::EmissionMap>(
					material.GetEmissionMap()->GetName() + "_emission");
				auto path = SaveMap<World::ObjectType::EmissionMap>(
					material.GetEmissionMap()->GetBitmap(), dir_path, name);
				map_names.add<World::ObjectType::EmissionMap>(material.GetEmissionMap(), std::move(name), std::move(path));
			}
		}

		try
		{
			std::ofstream file(mtllib_path);
			RZAssert(file.is_open(), "failed to save materials to " + mtllib_path.string());
			file.exceptions(file.failbit);

			for (const auto& [material_ref, material_name] : materials)
			{
				const auto& material = material_ref.get();

				SaveMaterial(material, file,
					material_name,
					MapsPaths{
					material.GetTexture() ? map_names.path<World::ObjectType::Texture>(material.GetTexture()) : "",
					material.GetNormalMap() ? map_names.path<World::ObjectType::NormalMap>(material.GetNormalMap()) : "",
					material.GetMetalnessMap() ? map_names.path<World::ObjectType::MetalnessMap>(material.GetMetalnessMap()) : "",
					material.GetRoughnessMap() ? map_names.path<World::ObjectType::RoughnessMap>(material.GetRoughnessMap()) : "",
					material.GetEmissionMap() ? map_names.path<World::ObjectType::EmissionMap>(material.GetEmissionMap()) : ""});
				file << std::endl;
			}
		}
		catch (std::system_error&)
		{
			std::filesystem::remove(mtllib_path);
			throw;
		}
		return mtllib_path;
	}
	std::filesystem::path MTLSaver::SaveMTLWithMaps(
		const Material& material,
		const std::filesystem::path& path,
		const std::string& file_name)
	{
		MapsPaths paths;
		if (const auto& map = material.GetTexture(); map)
			paths.texture = SaveMap<World::ObjectType::Texture>(map->GetBitmap(), path, file_name + "_color");
		if (const auto& map = material.GetNormalMap(); map)
			paths.normal = SaveMap<World::ObjectType::Texture>(map->GetBitmap(), path, file_name + "_normal");
		if (const auto& map = material.GetMetalnessMap(); map)
			paths.metalness = SaveMap<World::ObjectType::MetalnessMap>(map->GetBitmap(), path, file_name + "_metalness");
		if (const auto& map = material.GetRoughnessMap(); map)
			paths.roughness = SaveMap<World::ObjectType::RoughnessMap>(map->GetBitmap(), path, file_name + "_roughness");
		if (const auto& map = material.GetEmissionMap(); map)
			paths.normal = SaveMap<World::ObjectType::EmissionMap>(map->GetBitmap(), path, file_name + "_emission");

		return SaveMTL(
			material, path,
			file_name,
			paths);
	}
	void MTLSaver::SaveMaterial(
		const Material& material,
		std::ofstream& file,
		const std::string& name,
		const MapsPaths& maps_paths)
	{
		file << "newmtl " << name << '\n';
		// color (RGB)
		constexpr auto max = float(std::numeric_limits<decltype(material.GetColor().red)>::max());
		file
			<< "Kd "
			<< material.GetColor().red / max << ' '
			<< material.GetColor().green / max << ' '
			<< material.GetColor().blue / max << '\n';
		// alpha
		file << "d " << material.GetColor().alpha / max << '\n';
		// metalness
		file << "Pm " << material.GetMetalness() << '\n';
		// roughness
		file << "Pr " << material.GetRoughness() << '\n';
		// emission
		file << "Ke " << material.GetEmission() << '\n';
		// IOR
		file << "Ni " << material.GetIOR() << '\n';

		if (const auto& texture = material.GetTexture(); texture && !maps_paths.texture.empty())
			file << "map_Kd " << maps_paths.texture.string() << '\n';
		if (const auto& normal_map = material.GetNormalMap(); normal_map && !maps_paths.normal.empty())
			file << "norm " << maps_paths.normal.string() << '\n';
		if (const auto& metalness_map = material.GetMetalnessMap(); metalness_map && !maps_paths.metalness.empty())
			file << "map_Pm " << maps_paths.metalness.string() << '\n';
		if (const auto& roughness_map = material.GetRoughnessMap(); roughness_map && !maps_paths.roughness.empty())
			file << "map_Pr " << maps_paths.roughness.string() << '\n';
		if (const auto& emission_map = material.GetEmissionMap(); emission_map && !maps_paths.emission.empty())
			file << "map_Ke " << maps_paths.emission.string() << '\n';
	}


	std::filesystem::path OBJSaver::SaveOBJ(
		const MeshStructure& mesh,
		const std::filesystem::path& path,
		const std::optional<std::filesystem::path>& material_library,
		const std::unordered_map<uint32_t, std::string>& material_names)
	{
		RZAssert(path.has_filename(), "path must contain file name.obj");
		RZAssert(
			path.has_extension() && path.extension() == ".obj",
			"path must contain file name with .obj extension");

		try
		{
			if (const auto parent_path = path.parent_path(); !std::filesystem::exists(parent_path))
				std::filesystem::create_directories(parent_path);

			std::ofstream file(path);
			RZAssert(file.is_open(), "failed to save meshes to file " + path.string());
			file.exceptions(file.failbit);

			if (material_library)
				file << "mtllib " << material_library->string() << std::endl;

			SaveMesh(mesh, file, material_names);
		}
		catch (std::system_error&)
		{
			std::filesystem::remove(path);
			throw;
		}

		return path;
	}
	std::filesystem::path OBJSaver::SaveOBJ(
		const std::vector<Handle<Mesh>>& instances,
		const std::filesystem::path& path)
	{
		RZAssert(path.has_filename(), "path must contain file name.obj");
		RZAssert(
			path.has_extension() && path.extension() == ".obj",
			"path must contain file name with .obj extension");

		if (const auto parent_path = path.parent_path(); !std::filesystem::exists(parent_path))
			std::filesystem::create_directories(parent_path);

		// save materials
		ObjectNames<World::ObjectType::Material> material_names;
		std::vector<std::pair<std::reference_wrapper<Material>, std::string>> materials;
		for (const auto& instance : instances)
		{
			if (!instance) continue;
			for (uint32_t i = 0; i < instance->GetMaterialCapacity(); i++)
			{
				const auto& material = instance->GetMaterial(i);
				if (!material) continue;
				if (!material_names.contains<World::ObjectType::Material>(material))
				{
					auto unique_name = material_names.uniqueName<World::ObjectType::Material>(material->GetName());
					materials.push_back({std::ref(*material), unique_name});
					material_names.add<World::ObjectType::Material>(material, std::move(unique_name), "");
				}
			}
		}
		auto materials_path = SaveMTL(materials, std::filesystem::path{path}.remove_filename(), "materials");

		try
		{
			std::ofstream file(path);
			RZAssert(file.is_open(), "failed to open file " + path.string());
			file.exceptions(file.failbit);

			file << "mtllib " << OBJSaver::relative_path(path, materials_path).string() << '\n';

			// save meshes
			Math::vec3u32 ids_offsets(0u, 0u, 0u);
			for (const auto& instance : instances)
			{
				if (!instance) continue;
				if (const auto& mesh = instance->GetStructure(); mesh)
				{
					// collect material names for material ids
					std::unordered_map<uint32_t, std::string> material_name_map;
					for (uint32_t i = 0; i < instance->GetMaterialCapacity(); i++)
					{
						const auto& material = instance->GetMaterial(i);
						if (!material) continue;
						material_name_map[i] = material_names.name<World::ObjectType::Material>(material);
					}

					// save mesh
					SaveMesh(*mesh, file, material_name_map, ids_offsets);
					ids_offsets.x += mesh->GetVertices().GetCount();
					ids_offsets.y += mesh->GetTexcrds().GetCount();
					ids_offsets.z += mesh->GetNormals().GetCount();
				}
			}
		}
		catch (std::system_error&)
		{
			std::filesystem::remove(path);
			throw;
		}

		return path;
	}
	void OBJSaver::SaveMesh(
		const MeshStructure& mesh,
		std::ofstream& file,
		const std::unordered_map<uint32_t, std::string>& material_names,
		const Math::vec3u32& offsets)
	{
		file << "\ng " << mesh.GetName() << '\n';

		// write all vertices
		const auto& vertices = mesh.GetVertices();
		for (uint32_t i = 0; i < vertices.GetCount(); i++)
			file << "v " << vertices[i].x << ' ' << vertices[i].y << ' ' << -vertices[i].z << '\n';

		// write all texture coordinates
		const auto& texcrds = mesh.GetTexcrds();
		for (uint32_t i = 0; i < texcrds.GetCount(); i++)
			file << "vt " << texcrds[i].x << ' ' << texcrds[i].y << std::endl;

		// write all normals
		const auto& normals = mesh.GetNormals();
		for (uint32_t i = 0; i < normals.GetCount(); i++)
			file << "vn " << normals[i].x << ' ' << normals[i].y << ' ' << -normals[i].z << '\n';

		// write all triangles (faces)
		const auto& triangles = mesh.GetTriangles();
		uint32_t current_material_idx = material_names.empty() ? 0 : std::numeric_limits<uint32_t>::max();
		for (uint32_t i = 0; i < triangles.GetCount(); i++)
		{
			const auto& triangle = triangles[i];
			if (current_material_idx != triangle.material_id)
			{
				current_material_idx = triangle.material_id;
				if (material_names.empty())
				{
					// no material names provided, save material name as index, to preserve multi-material properties
					// of the mesh
					file << "usemtl " << current_material_idx << '\n';
				}
				else
				{
					// check if name for the index has been provided
					if (const auto entry = material_names.find(current_material_idx); entry != material_names.end())
					{
						file << "usemtl " << entry->second << '\n';
					}
				}
			}

			const auto print_ids = [&file, &offsets](const uint32_t& v, const uint32_t& t, const uint32_t& n)
			{
				static constexpr auto unused = ComponentContainer<Vertex>::sm_npos;
				file << ' ' << v + 1 + offsets.x;
				if (t != unused)
					file << '/' << t + 1 + offsets.y;
				if (n != unused)
					file << (t == unused ? "//" : "/") << n + 1 + offsets.z;
			};

			// 0, 2, 1 - to translate from RZ clockwise to .mtl format anti-clockwise order
			file << "f";
			print_ids(triangle.vertices[0], triangle.texcrds[0], triangle.normals[0]);
			print_ids(triangle.vertices[2], triangle.texcrds[2], triangle.normals[2]);
			print_ids(triangle.vertices[1], triangle.texcrds[1], triangle.normals[1]);
			file << '\n';
		}
	}


	Saver::Saver(World& world)
		: OBJSaver(world)
		, mp_json_saver(new JsonSaver(world))
	{}
	void Saver::SaveScene(const SaveOptions& options)
	{
		try
		{
			mp_json_saver->saveJsonScene(options);
		}
		catch (std::filesystem::filesystem_error&)
		{
			if (!options.allow_partial_write)
			{
				// TODO: when this throws, add info, that it failed during previous fail 
				if (options.path.has_filename())
					std::filesystem::remove_all(options.path.parent_path());
				else
					std::filesystem::remove_all(options.path);
			}
			throw;
		}
	}
}
