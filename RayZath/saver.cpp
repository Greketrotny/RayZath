#include "saver.h"
#include "json_saver.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "./lib/stb_image/stb_image_write.h"

#include <fstream>

namespace RayZath::Engine
{
	using namespace std::string_literals;
	/*void BitmapSaver::SaveAllMaps(const std::filesystem::path& path)
	{
		SaveAllTypeMaps<World::ObjectType::Texture>(path / "textures");
		SaveAllTypeMaps<World::ObjectType::NormalMap>(path / "normal_maps");
		SaveAllTypeMaps<World::ObjectType::MetalnessMap>(path / "metalness_maps");
		SaveAllTypeMaps<World::ObjectType::RoughnessMap>(path / "roughness_maps");
		SaveAllTypeMaps<World::ObjectType::EmissionMap>(path / "emission_maps");
	}*/
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
			map.GetWidth(), map.GetHeight(),
			4,
			map.GetMapAddress(),
			map.GetWidth() * sizeof(*map.GetMapAddress())),
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
			map.GetWidth(), map.GetHeight(),
			1,
			map.GetMapAddress(),
			map.GetWidth() * sizeof(*map.GetMapAddress())),
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
			map.GetWidth(), map.GetHeight(),
			1,
			map.GetMapAddress()),
			"failed to write image file to "s + full_path.string());
		return full_path;
	}


	std::filesystem::path MTLSaver::SaveMTL(
		const Handle<Material>& material,
		const std::filesystem::path& path,
		const std::string& file_name,
		const MapsPaths& maps_paths)
	{
		RZAssert(material, "tried to save invalid material");

		if (!std::filesystem::exists(path))
			std::filesystem::create_directories(path);
		const auto full_path = path / (file_name + ".mtl");
		try
		{
			std::ofstream file(full_path);
			RZAssert(file.is_open(), "failed to save material " + material->GetName() + " to " + full_path.string());
			file.exceptions(file.failbit);

			file << "newmtl " << file_name << "\n\n";
			// color (RGB)
			constexpr auto max = float(std::numeric_limits<decltype(material->GetColor().red)>::max());
			file
				<< "Kd "
				<< material->GetColor().red / max << ' '
				<< material->GetColor().green / max << ' '
				<< material->GetColor().blue / max << '\n';
			// color (A)
			file << "d " << material->GetColor().alpha / max << std::endl;
			// metalness
			file << "Pm " << material->GetMetalness() << std::endl;
			// roughness
			file << "Pr " << material->GetRoughness() << std::endl;
			// emission
			file << "Ke " << material->GetEmission() << std::endl;
			// IOR
			file << "Ni " << material->GetIOR() << std::endl;

			file << std::endl;

			if (const auto& texture = material->GetTexture(); texture && !maps_paths.texture.empty())
				file << "map_Kd " << maps_paths.texture << std::endl;
			if (const auto& normal_map = material->GetNormalMap(); normal_map && !maps_paths.normal.empty())
				file << "norm " << maps_paths.normal << std::endl;
			if (const auto& metalness_map = material->GetMetalnessMap(); metalness_map && !maps_paths.metalness.empty())
				file << "map_Pm " << maps_paths.metalness << std::endl;
			if (const auto& roughness_map = material->GetRoughnessMap(); roughness_map && !maps_paths.roughness.empty())
				file << "map_Pr " << maps_paths.roughness << std::endl;
			if (const auto& emission_map = material->GetEmissionMap(); emission_map && !maps_paths.emission.empty())
				file << "map_Ke " << maps_paths.emission << std::endl;
		}
		catch (std::system_error&)
		{
			std::filesystem::remove(full_path);
			throw;
		}
		return full_path;
	}
	std::filesystem::path MTLSaver::SaveMTLWithMaps(
		const Handle<Material>& material,
		const std::filesystem::path& path,
		const std::string& file_name)
	{
		MapsPaths paths;
		if (const auto& map = material->GetTexture(); map)
			paths.texture = SaveMap<World::ObjectType::Texture>(map->GetBitmap(), path, material->GetName() + "_color");
		if (const auto& map = material->GetNormalMap(); map)
			paths.normal = SaveMap<World::ObjectType::Texture>(map->GetBitmap(), path, material->GetName() + "_normal");
		if (const auto& map = material->GetMetalnessMap(); map)
			paths.metalness = SaveMap<World::ObjectType::MetalnessMap>(map->GetBitmap(), path, material->GetName() + "_metalness");
		if (const auto& map = material->GetRoughnessMap(); map)
			paths.roughness = SaveMap<World::ObjectType::RoughnessMap>(map->GetBitmap(), path, material->GetName() + "_roughness");
		if (const auto& map = material->GetEmissionMap(); map)
			paths.normal = SaveMap<World::ObjectType::EmissionMap>(map->GetBitmap(), path, material->GetName() + "_emission");

		return SaveMTL(
			material, path,
			material->GetName(),
			paths);
	}


	void OBJSaver::SaveOBJ(
		const std::vector<Handle<MeshStructure>>& meshes,
		const std::filesystem::path& path,
		const std::optional<std::filesystem::path>& material_library,
		const std::unordered_map<uint32_t, std::string>& material_names)
	{
		try
		{
			std::ofstream file(path);
			RZAssert(file.is_open(), "failed to save meshes to file " + path.string());
			file.exceptions(file.failbit);

			if (material_library)
				file << "mtllib " << *material_library << std::endl;

			for (const auto& mesh : meshes)
				SaveMesh(mesh, file, material_names);
		}
		catch (std::system_error&)
		{
			std::filesystem::remove(path);
			throw;
		}
	}
	void OBJSaver::SaveMesh(
		const Handle<MeshStructure>& mesh,
		std::ofstream& file,
		const std::unordered_map<uint32_t, std::string>& material_names)
	{
		if (!mesh) return;

		file << "\ng " << mesh->GetName() << '\n';

		const auto& vertices = mesh->GetVertices();
		for (uint32_t i = 0; i < vertices.GetCount(); i++)
			file << "v " << vertices[i].x << ' ' << vertices[i].y << ' ' << -vertices[i].z << '\n';
		const auto& texcrds = mesh->GetTexcrds();
		for (uint32_t i = 0; i < texcrds.GetCount(); i++)
			file << "vt " << texcrds[i].x << ' ' << texcrds[i].y << std::endl;
		const auto& normals = mesh->GetNormals();
		for (uint32_t i = 0; i < normals.GetCount(); i++)
			file << "vn " << normals[i].x << ' ' << normals[i].y << ' ' << -normals[i].z << '\n';

		const auto& triangles = mesh->GetTriangles();
		uint32_t current_material_idx = material_names.empty() ? 0 : std::numeric_limits<uint32_t>::max();
		for (uint32_t i = 0; i < triangles.GetCount(); i++)
		{
			const auto& triangle = triangles[i];
			if (current_material_idx != triangle.material_id)
			{
				current_material_idx = triangle.material_id;
				file << "usemtl " << material_names.at(current_material_idx) << '\n';
			}

			const auto print_ids = [&file](const uint32_t& v, const uint32_t& t, const uint32_t& n)
			{
				static constexpr auto unused = ComponentContainer<Vertex>::sm_npos;
				file << ' ' << v + 1;
				if (t != unused)
					file << '/' << t + 1;
				if (n != unused)
					file << (t == unused ? "//" : "/") << n + 1;
			};

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
				std::filesystem::remove_all(options.path);
			}
			throw;
		}
	}
}
