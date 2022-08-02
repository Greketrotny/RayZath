#include "saver.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "./lib/stb_image/stb_image_write.h"


namespace RayZath::Engine
{
	void BitmapSaver::SaveAllMaps(const std::filesystem::path& path)
	{
		SaveAllTypeMaps<World::ObjectType::Texture>(path / "textures");
		SaveAllTypeMaps<World::ObjectType::NormalMap>(path / "normal_maps");
		SaveAllTypeMaps<World::ObjectType::MetalnessMap>(path / "metalness_maps");
		SaveAllTypeMaps<World::ObjectType::RoughnessMap>(path / "roughness_maps");
		SaveAllTypeMaps<World::ObjectType::EmissionMap>(path / "emission_maps");
	}
	template<>
	void BitmapSaver::SaveMap<World::ObjectType::Texture>(
		const Graphics::Bitmap& map,
		const std::filesystem::path& path,
		const std::string& file_name)
	{
		RZAssert(0 != stbi_write_png(
			(path / (file_name + ".jpg")).string().c_str(),
			map.GetWidth(), map.GetHeight(),
			4,
			map.GetMapAddress(),
			map.GetWidth() * sizeof(*map.GetMapAddress())),
			"failed to write image file");
	}
	template <>
	void BitmapSaver::SaveMap<World::ObjectType::NormalMap>(
		const Graphics::Bitmap& map,
		const std::filesystem::path& path,
		const std::string& file_name)
	{
		SaveMap<World::ObjectType::Texture>(map, path, file_name);
	}
	template <>
	void BitmapSaver::SaveMap<World::ObjectType::MetalnessMap>(
		const Graphics::Buffer2D<uint8_t>& map,
		const std::filesystem::path& path,
		const std::string& file_name)
	{
		RZAssert(0 != stbi_write_png(
			(path / (file_name + ".jpg")).string().c_str(),
			map.GetWidth(), map.GetHeight(),
			1,
			map.GetMapAddress(),
			map.GetWidth() * sizeof(*map.GetMapAddress())),
			"failed to write image file");
	}
	template <>
	void BitmapSaver::SaveMap<World::ObjectType::RoughnessMap>(
		const Graphics::Buffer2D<uint8_t>& map,
		const std::filesystem::path& path,
		const std::string& file_name)
	{
		SaveMap<World::ObjectType::MetalnessMap>(map, path, file_name);
	}
	template <>
	void BitmapSaver::SaveMap<World::ObjectType::EmissionMap>(
		const Graphics::Buffer2D<float>& map,
		const std::filesystem::path& path,
		const std::string& file_name)
	{
		RZAssert(0 != stbi_write_hdr(
			(path / (file_name + ".jpg")).string().c_str(),
			map.GetWidth(), map.GetHeight(),
			1,
			map.GetMapAddress()),
			"failed to write image file");
	}
	
	

	Saver::Saver(World& world)
		: BitmapSaver(world)
	{}

	void Saver::SaveScene(const SaveOptions& options)
	{
		try
		{
			const auto maps_path = options.path / "maps";
			std::filesystem::create_directory(maps_path);
			SaveAllMaps(maps_path);
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
