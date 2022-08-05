#include "saver.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "./lib/stb_image/stb_image_write.h"

#include <fstream>

namespace RayZath::Engine
{
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
		std::filesystem::path full_path = path / (file_name + ".png");
		RZAssert(0 != stbi_write_png(
			full_path.string().c_str(),
			map.GetWidth(), map.GetHeight(),
			4,
			map.GetMapAddress(),
			map.GetWidth() * sizeof(*map.GetMapAddress())),
			"failed to write image file");
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
		std::filesystem::path full_path = path / (file_name + ".jpg");
		RZAssert(0 != stbi_write_png(
			full_path.string().c_str(),
			map.GetWidth(), map.GetHeight(),
			1,
			map.GetMapAddress(),
			map.GetWidth() * sizeof(*map.GetMapAddress())),
			"failed to write image file");
		return full_path;
	}
	template <>
	std::filesystem::path BitmapSaver::SaveMap<World::ObjectType::RoughnessMap>(
		const Graphics::Buffer2D<uint8_t>& map,
		const std::filesystem::path& path,
		const std::string& file_name)
	{
		return SaveMap<World::ObjectType::MetalnessMap>(map, path, file_name);
	}
	template <>
	std::filesystem::path BitmapSaver::SaveMap<World::ObjectType::EmissionMap>(
		const Graphics::Buffer2D<float>& map,
		const std::filesystem::path& path,
		const std::string& file_name)
	{
		std::filesystem::path full_path = path / (file_name + ".jpg");
		RZAssert(0 != stbi_write_hdr(
			full_path.string().c_str(),
			map.GetWidth(), map.GetHeight(),
			1,
			map.GetMapAddress()),
			"failed to write image file");
		return full_path;
	}
	

	void MTLSaver::SaveMTL(
		const Handle<Material>& material, 
		const std::filesystem::path& path,
		const std::optional<MapsPaths>& maps_paths,
		const std::string& file_name)
	{
		if (!material) return;

		const auto full_path = path / file_name / ".mtl";
		try
		{
			std::ofstream file(full_path);
			RZAssert(file.is_open(), "failed to save material " + material->GetName() + " to " + full_path.string());
			file.exceptions(file.failbit);

			file << "newmtl " << file_name << std::endl;
			// color (RGB)
			file
				<< "Kd " 
				<< material->GetColor().red 
				<< material->GetColor().green
				<< material->GetColor().blue
				<< std::endl;
			// color (A)
			file << "d " << material->GetColor().alpha << std::endl;
			// metalness
			file << "Pm " << material->GetMetalness() << std::endl;
			// roughness
			file << "Pr " << material->GetRoughness() << std::endl;
			// emission
			file << "Ke " << material->GetEmission() << std::endl;
			// IOR
			file << "Ni " << material->GetIOR() << std::endl;

			if (maps_paths)
			{
				if (const auto& texture = material->GetTexture(); texture)
					file << "map_Kd " << maps_paths->texture << std::endl;
				if (const auto& normal_map = material->GetNormalMap(); normal_map)
					file << "norm " << maps_paths->normal << std::endl;
				if (const auto& metalness_map = material->GetMetalnessMap(); metalness_map)
					file << "map_Pm " << maps_paths->metalness << std::endl;
				if (const auto& roughness_map = material->GetRoughnessMap(); roughness_map)
					file << "map_Pr " << maps_paths->roughness << std::endl;
				if (const auto& emission_map = material->GetEmissionMap(); emission_map)
					file << "map_Ke " << maps_paths->emission << std::endl;
			}
		}
		catch (std::system_error&)
		{
			std::filesystem::remove(full_path);
			throw;
		}
	}

	void Saver::SaveScene(const SaveOptions& options)
	{
		try
		{
			/*m_root_path = options.path;
			m_names.reset();
			const auto maps_path = options.path / "maps";
			std::filesystem::create_directory(maps_path);
			SaveAllMaps(maps_path);*/
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
