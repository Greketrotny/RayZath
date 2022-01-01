#include "json_loader.h"

#include "loader.h"

namespace RayZath::Engine
{
	JsonLoader::JsonLoader(World& world)
		: mr_world(world)
	{}

	std::filesystem::path JsonLoader::ModifyPath(std::filesystem::path path)
	{
		if (path.is_relative())
			path = m_path.parent_path() / path;

		return path;
	}

	template <typename T, std::enable_if_t<std::is_same_v<T, Math::vec3f>, bool> = false>
	T JsonTo(const nlohmann::json& vec3_json)
	{
		if (!vec3_json.is_array()) throw Exception("Value is not an array.");
		if (vec3_json.size() != 3u) throw Exception("Array has to have three coordinates.");
		if (!(vec3_json[0].is_number() &&
			vec3_json[1].is_number() &&
			vec3_json[2].is_number())) throw Exception("Coordinates should be numbers.");

		auto values = vec3_json.get<std::array<float, 3u>>();
		return T(values[0], values[1], values[2]);
	}
	template <typename T, std::enable_if_t<std::is_same_v<T, Math::vec2f> || std::is_same_v<T, Math::vec2ui32>, bool> = false>
	T JsonTo(const nlohmann::json& vec2_json)
	{
		if (!vec2_json.is_array()) throw Exception("Value is not an array.");
		if (vec2_json.size() != 2u) throw Exception("Array has to have two coordinates.");
		if (!(vec2_json[0].is_number() &&
			vec2_json[1].is_number())) throw Exception("Coordinates should be numbers.");

		auto values = vec2_json.get<std::array<float, 2u>>();
		return T(values[0], values[1]);
	}
	template <typename T, std::enable_if_t<std::is_same_v<T, Graphics::Color>, bool> = false>
	T JsonTo(const nlohmann::json& json)
	{
		if (!json.is_array()) throw Exception("Value is not an array.");
		if (json.size() < 3u) throw Exception("Color has at least three channels.");

		std::array<uint8_t, 4u> values = { 0xF0, 0xF0, 0xF0, 0xFF };
		for (size_t i = 0u; i < json.size(); i++)
		{
			if (!json[i].is_number()) throw Exception("Color values should be numbers.");
			if (json[i].is_number_float())
				values[i] = uint8_t(std::clamp(float(json[i]), 0.0f, 1.0f) * 255.0f);
			else if (json[i].is_number_integer())
				values[i] = uint8_t(std::clamp<uint32_t>(uint32_t(json[i]), 0u, 255u));
		}

		return T(values[0], values[1], values[2], values[3]);
	}


	template<> Handle<Texture> JsonLoader::Load<World::ContainerType::Texture>(const nlohmann::json& json)
	{
		if (json.is_string())
		{
			return mr_world.Container<World::ContainerType::Texture>()
				[static_cast<std::string>(json)];
		}
		else if (json.is_object())
		{
			ConStruct<Texture> construct;
			for (auto& item : json.items())
			{
				auto& key = item.key();
				auto& value = item.value();

				if (key == "name" && value.is_string())
					construct.name = value;
				else if (key == "filter mode" && value.is_string())
				{
					if (value == "point") construct.filter_mode = Texture::FilterMode::Point;
					else if (value == "linear") construct.filter_mode = Texture::FilterMode::Linear;
				}
				else if (key == "address mode" && value.is_string())
				{
					if (value == "wrap") construct.address_mode = Texture::AddressMode::Wrap;
					else if (value == "clamp") construct.address_mode = Texture::AddressMode::Clamp;
					else if (value == "mirror") construct.address_mode = Texture::AddressMode::Mirror;
					else if (value == "border") construct.address_mode = Texture::AddressMode::Border;
				}
				else if (key == "scale" && value.is_array())
					construct.scale = JsonTo<Math::vec2f>(value);
				else if (key == "rotation" && value.is_number())
					construct.rotation = value;
				else if (key == "translation" && value.is_array())
					construct.translation = JsonTo<Math::vec2f>(value);
				else if (key == "file" && value.is_string())
					construct.bitmap = mr_world.GetLoader().LoadTexture(
						ModifyPath(static_cast<std::string>(value)).string());
			}

			return mr_world.Container<World::ContainerType::Texture>().Create(construct);
		}

		return {};
	}
	template<> Handle<NormalMap> JsonLoader::Load<World::ContainerType::NormalMap>(const nlohmann::json& json)
	{
		if (json.is_string())
		{
			return mr_world.Container<World::ContainerType::NormalMap>()
				[static_cast<std::string>(json)];
		}
		else if (json.is_object())
		{
			ConStruct<NormalMap> construct;
			for (auto& item : json.items())
			{
				auto& key = item.key();
				auto& value = item.value();

				if (key == "name" && value.is_string())
					construct.name = value;
				else if (key == "filter mode" && value.is_string())
				{
					if (value == "point") construct.filter_mode = NormalMap::FilterMode::Point;
					else if (value == "linear") construct.filter_mode = NormalMap::FilterMode::Linear;
				}
				else if (key == "address mode" && value.is_string())
				{
					if (value == "wrap") construct.address_mode = NormalMap::AddressMode::Wrap;
					else if (value == "clamp") construct.address_mode = NormalMap::AddressMode::Clamp;
					else if (value == "mirror") construct.address_mode = NormalMap::AddressMode::Mirror;
					else if (value == "border") construct.address_mode = NormalMap::AddressMode::Border;
				}
				else if (key == "scale" && value.is_array())
					construct.scale = JsonTo<Math::vec2f>(value);
				else if (key == "rotation" && value.is_number())
					construct.rotation = value;
				else if (key == "translation" && value.is_array())
					construct.translation = JsonTo<Math::vec2f>(value);
				else if (key == "file" && value.is_string())
					construct.bitmap = mr_world.GetLoader().LoadNormalMap(
						ModifyPath(static_cast<std::string>(value)).string());
			}

			return mr_world.Container<World::ContainerType::NormalMap>().Create(construct);
		}

		return {};
	}
	template<> Handle<MetalnessMap> JsonLoader::Load<World::ContainerType::MetalnessMap>(const nlohmann::json& json)
	{
		if (json.is_string())
		{
			return mr_world.Container<World::ContainerType::MetalnessMap>()
				[static_cast<std::string>(json)];
		}
		else if (json.is_object())
		{
			ConStruct<MetalnessMap> construct;
			for (auto& item : json.items())
			{
				auto& key = item.key();
				auto& value = item.value();

				if (key == "name" && value.is_string())
					construct.name = value;
				else if (key == "filter mode" && value.is_string())
				{
					if (value == "point") construct.filter_mode = MetalnessMap::FilterMode::Point;
					else if (value == "linear") construct.filter_mode = MetalnessMap::FilterMode::Linear;
				}
				else if (key == "address mode" && value.is_string())
				{
					if (value == "wrap") construct.address_mode = MetalnessMap::AddressMode::Wrap;
					else if (value == "clamp") construct.address_mode = MetalnessMap::AddressMode::Clamp;
					else if (value == "mirror") construct.address_mode = MetalnessMap::AddressMode::Mirror;
					else if (value == "border") construct.address_mode = MetalnessMap::AddressMode::Border;
				}
				else if (key == "scale" && value.is_array())
					construct.scale = JsonTo<Math::vec2f>(value);
				else if (key == "rotation" && value.is_number())
					construct.rotation = value;
				else if (key == "translation" && value.is_array())
					construct.translation = JsonTo<Math::vec2f>(value);
				else if (key == "file" && value.is_string())
					construct.bitmap = mr_world.GetLoader().LoadMetalnessMap(
						ModifyPath(static_cast<std::string>(value)).string());
			}

			return mr_world.Container<World::ContainerType::MetalnessMap>().Create(construct);
		}

		return {};
	}
	template<> Handle<RoughnessMap> JsonLoader::Load<World::ContainerType::RoughnessMap>(const nlohmann::json& json)
	{
		if (json.is_string())
		{
			return mr_world.Container<World::ContainerType::RoughnessMap>()
				[static_cast<std::string>(json)];
		}
		else if (json.is_object())
		{
			ConStruct<RoughnessMap> construct;
			for (auto& item : json.items())
			{
				auto& key = item.key();
				auto& value = item.value();

				if (key == "name" && value.is_string())
					construct.name = value;
				else if (key == "filter mode" && value.is_string())
				{
					if (value == "point") construct.filter_mode = RoughnessMap::FilterMode::Point;
					else if (value == "linear") construct.filter_mode = RoughnessMap::FilterMode::Linear;
				}
				else if (key == "address mode" && value.is_string())
				{
					if (value == "wrap") construct.address_mode = RoughnessMap::AddressMode::Wrap;
					else if (value == "clamp") construct.address_mode = RoughnessMap::AddressMode::Clamp;
					else if (value == "mirror") construct.address_mode = RoughnessMap::AddressMode::Mirror;
					else if (value == "border") construct.address_mode = RoughnessMap::AddressMode::Border;
				}
				else if (key == "scale" && value.is_array())
					construct.scale = JsonTo<Math::vec2f>(value);
				else if (key == "rotation" && value.is_number())
					construct.rotation = value;
				else if (key == "translation" && value.is_array())
					construct.translation = JsonTo<Math::vec2f>(value);
				else if (key == "file" && value.is_string())
					construct.bitmap = mr_world.GetLoader().LoadRoughnessMap(
						ModifyPath(static_cast<std::string>(value)).string());
			}

			return mr_world.Container<World::ContainerType::RoughnessMap>().Create(construct);
		}

		return {};
	}
	template<> Handle<EmissionMap> JsonLoader::Load<World::ContainerType::EmissionMap>(const nlohmann::json& json)
	{
		// TODO: add emission map loading
		ThrowException("Load<EmissionMap>() not implemented.");
		return {};
	}

	template<> Handle<Material> JsonLoader::Load<World::ContainerType::Material>(const nlohmann::json& json)
	{
		if (json.is_string())
		{
			return mr_world.Container<World::ContainerType::Material>()
				[static_cast<std::string>(json)];
		}
		else if (json.is_object())
		{
			if (json.contains("file"))
			{
				auto& value = json["file"];
				if (!value.is_string())
					throw Exception("Path to file must be string.");

				auto materials = mr_world.GetLoader().LoadMTL(
					ModifyPath(static_cast<std::string>(value)).string());
				if (materials.empty())
					throw Exception("Failed to load any materials from file: " + value);

				return *materials.begin();
			}

			ConStruct<Material> construct;
			for (auto& item : json.items())
			{
				auto& key = item.key();
				auto& value = item.value();

				if (key == "name" && value.is_string())
					construct.name = value;
				else if (key == "color")
					construct.color = JsonTo<Graphics::Color>(value);
				else if (key == "metalness" && value.is_number())
					construct.metalness = std::clamp(float(value), 0.0f, 1.0f);
				else if (key == "roughness" && value.is_number())
					construct.roughness = std::clamp(float(value), 0.0f, 1.0f);
				else if (key == "emission" && value.is_number())
					construct.emission = std::clamp(float(value), 0.0f, std::numeric_limits<float>::infinity());
				else if (key == "ior" && value.is_number())
					construct.ior = std::clamp(float(value), 1.0f, std::numeric_limits<float>::infinity());
				else if (key == "scattering" && value.is_number())
					construct.scattering = std::clamp(float(value), 0.0f, std::numeric_limits<float>::infinity());

				else if (key == "texture")
					construct.texture = Load<World::ContainerType::Texture>(value);
				else if (key == "normal map")
					construct.normal_map = Load<World::ContainerType::NormalMap>(value);
				else if (key == "metalness map")
					construct.metalness_map = Load<World::ContainerType::MetalnessMap>(value);
				else if (key == "roughness map")
					construct.roughness_map = Load<World::ContainerType::RoughnessMap>(value);
				else if (key == "emission map")
					construct.emission_map = Load<World::ContainerType::EmissionMap>(value);
			}

			return mr_world.Container<World::ContainerType::Material>().Create(construct);
		}

		return {};
	}
	template<> Handle<MeshStructure> JsonLoader::Load<World::ContainerType::MeshStructure>(const nlohmann::json& json)
	{
		if (json.is_string())
		{
			return mr_world.Container<World::ContainerType::MeshStructure>()
				[static_cast<std::string>(json)];
		}
		else if (json.is_object())
		{
			if (!json.contains("name"))
				throw Exception("MeshStructure has to have name specified.");

			std::vector<const nlohmann::json*> vertices, texcrds, normals, triangles;
			ConStruct<MeshStructure> construct;
			for (auto& item : json.items())
			{
				auto& key = item.key();
				auto& value = item.value();

				if (key == "name" && value.is_string())
					construct.name = value;
				if (key == "vertices" && value.is_array())
					vertices.push_back(&value);
				else if (key == "texcrds" && value.is_array())
					texcrds.push_back(&value);
				else if (key == "normals" && value.is_array())
					normals.push_back(&value);
				else if (key == "triangles" && value.is_array())
					triangles.push_back(&value);
			}

			Handle<MeshStructure> structure =
				mr_world.Container<World::ContainerType::MeshStructure>().Create(construct);
			if (!structure)
				return structure;

			for (auto& vs : vertices)
				for (auto& v : *vs)
					structure->CreateVertex(JsonTo<Vertex>(v));

			for (auto& ts : texcrds)
				for (auto& t : *ts)
					structure->CreateTexcrd(JsonTo<Texcrd>(t));

			for (auto& ns : normals)
				for (auto& n : *ns)
					structure->CreateNormal(JsonTo<Normal>(n));

			for (auto& ts : triangles)
				for (auto& triangle : *ts)
				{
					if (!triangle.is_object())
						continue;

					std::array<std::array<uint32_t, 3u>, 3u> indices{};
					uint32_t material_idx = 0u;
					for (auto& v : indices)
					{
						v[0] = ComponentContainer<Vertex>::GetEndPos();
						v[1] = ComponentContainer<Texcrd>::GetEndPos();
						v[2] = ComponentContainer<Normal>::GetEndPos();
					}

					for (auto& component : triangle.items())
					{
						if (component.key() == "v")
							indices[0] = component.value().get<std::array<uint32_t, 3u>>();
						else if (component.key() == "t")
							indices[1] = component.value().get<std::array<uint32_t, 3u>>();
						else if (component.key() == "n")
							indices[2] = component.value().get<std::array<uint32_t, 3u>>();
						else if (component.key() == "m" && component.value().is_number_integer())
							material_idx = component.value();
					}

					structure->CreateTriangle(indices[0], indices[1], indices[2], material_idx);
				}

			return structure;
		}

		return {};
	}

	template<> Handle<Camera> JsonLoader::Load<World::ContainerType::Camera>(const nlohmann::json& camera_json)
	{
		ConStruct<Camera> construct;
		for (auto& item : camera_json.items())
		{
			auto& key = item.key();
			auto& value = item.value();

			if (key == "name" && value.is_string())
				construct.name = value;
			else if (key == "position")
				construct.position = JsonTo<Math::vec3f>(value);
			else if (key == "rotation")
				construct.rotation = JsonTo<Math::vec3f>(value);
			else if (key == "resolution")
				construct.resolution = JsonTo<Math::vec2ui32>(value);
			else if (key == "fov" && value.is_number())
				construct.fov = value;
			else if (key == "near plane" && value.is_number())
				construct.near_far.x = value;
			else if (key == "far plane" && value.is_number())
				construct.near_far.y = value;
			else if (key == "near far")
				construct.near_far = JsonTo<Math::vec2f>(value);
			else if (key == "focal distance" && value.is_number())
				construct.focal_distance = value;
			else if (key == "aperture" && value.is_number())
				construct.aperture = value;
			else if (key == "exposure time" && value.is_number())
				construct.exposure_time = value;
			else if (key == "temporal blend" && value.is_number())
				construct.temporal_blend = value;
			else if (key == "enabled" && value.is_boolean())
				construct.enabled = value;
		}

		return mr_world.Container<World::ContainerType::Camera>().Create(construct);
	}

	template<> Handle<PointLight> JsonLoader::Load<World::ContainerType::PointLight>(const nlohmann::json& json)
	{
		ConStruct<PointLight> construct;
		for (auto& item : json.items())
		{
			auto& key = item.key();
			auto& value = item.value();

			if (key == "name" && value.is_string())
				construct.name = value;
			else if (key == "position")
				construct.position = JsonTo<Math::vec3f>(value);
			else if (key == "color")
				construct.color = JsonTo<Graphics::Color>(value);
			else if (key == "size" && value.is_number())
				construct.size = value;
			else if (key == "emission" && value.is_number())
				construct.emission = value;
		}

		return mr_world.Container<World::ContainerType::PointLight>().Create(construct);
	}
	template<> Handle<SpotLight> JsonLoader::Load<World::ContainerType::SpotLight>(const nlohmann::json& json)
	{
		ConStruct<SpotLight> construct;
		for (auto& item : json.items())
		{
			auto& key = item.key();
			auto& value = item.value();

			if (key == "name" && value.is_string())
				construct.name = value;
			else if (key == "position")
				construct.position = JsonTo<Math::vec3f>(value);
			else if (key == "direction")
				construct.direction = JsonTo<Math::vec3f>(value);
			else if (key == "color")
				construct.color = JsonTo<Graphics::Color>(value);
			else if (key == "size" && value.is_number())
				construct.size = value;
			else if (key == "emission" && value.is_number())
				construct.emission = value;
			else if (key == "angle" && value.is_number())
				construct.beam_angle = value;
			else if (key == "sharpness" && value.is_number())
				construct.sharpness = value;
		}

		return mr_world.Container<World::ContainerType::SpotLight>().Create(construct);
	}
	template<> Handle<DirectLight> JsonLoader::Load<World::ContainerType::DirectLight>(const nlohmann::json& json)
	{
		ConStruct<DirectLight> construct;
		for (auto& item : json.items())
		{
			auto& key = item.key();
			auto& value = item.value();

			if (key == "name" && value.is_string())
				construct.name = value;
			else if (key == "direction")
				construct.direction = JsonTo<Math::vec3f>(value);
			else if (key == "color")
				construct.color = JsonTo<Graphics::Color>(value);
			else if (key == "emission" && value.is_number())
				construct.emission = value;
			else if (key == "size" && value.is_number())
				construct.angular_size = value;
		}

		return mr_world.Container<World::ContainerType::DirectLight>().Create(construct);
	}

	template<> Handle<Mesh> JsonLoader::Load<World::ContainerType::Mesh>(const nlohmann::json& json)
	{
		if (json.is_object())
		{
			if (json.contains("file"))
			{
				auto& value = json["file"];
				if (!value.is_string())
					throw Exception("Path to .obj. file should be string.");

				auto objects = mr_world.GetLoader().LoadOBJ(ModifyPath(static_cast<std::string>(value)));
				if (objects.empty())
					throw Exception("Failed to load any object from file: " + value);

				return *objects.begin();
			}

			ConStruct<Mesh> construct;
			uint32_t material_count = 0u;

			for (auto& item : json.items())
			{
				auto& key = item.key();
				auto& value = item.value();

				if (key == "name" && value.is_string())
					construct.name = value;
				else if (key == "position")
					construct.position = JsonTo<Math::vec3f>(value);
				else if (key == "rotation")
					construct.rotation = JsonTo<Math::vec3f>(value);
				else if (key == "center")
					construct.center = JsonTo<Math::vec3f>(value);
				else if (key == "scale")
					construct.scale = JsonTo<Math::vec3f>(value);
				else if (key == "Material")
				{
					if (value.is_object())
					{
						if (material_count < Mesh::GetMaterialCapacity())
							construct.material[material_count++] = 
							Load<World::ContainerType::Material>(value);
					}
					else if (value.is_array())
					{
						for (auto& m : value)
						{
							if (material_count < Mesh::GetMaterialCapacity())
								construct.material[material_count++] = 
								Load<World::ContainerType::Material>(m);
						}
					}
					else if (value.is_string())
					{
						if (material_count < Mesh::GetMaterialCapacity())
							construct.material[material_count++] =
							mr_world.Container<World::ContainerType::Material>()
							[static_cast<std::string>(value)];
					}
				}
				else if (key == "MeshStructure")
				{
					if (construct.mesh_structure)
						throw Exception("Mesh structure already defined.");

					construct.mesh_structure = 
						Load<World::ContainerType::MeshStructure>(value);
				}
			}

			return mr_world.Container<World::ContainerType::Mesh>().Create(construct);
		}

		return {};
	}
	template<> Handle<Sphere> JsonLoader::Load<World::ContainerType::Sphere>(const nlohmann::json& json)
	{
		ConStruct<Sphere> construct;
		for (auto& item : json.items())
		{
			auto& key = item.key();
			auto& value = item.value();

			if (key == "name" && value.is_string())
				construct.name = value;
			else if (key == "position")
				construct.position = JsonTo<Math::vec3f>(value);
			else if (key == "rotation")
				construct.rotation = JsonTo<Math::vec3f>(value);
			else if (key == "center")
				construct.center = JsonTo<Math::vec3f>(value);
			else if (key == "scale")
				construct.scale = JsonTo<Math::vec3f>(value);
			else if (key == "radius" && value.is_number())
				construct.radius = value;
			else if (key == "Material" && value.is_object())
				construct.material = Load<World::ContainerType::Material>(value);
		}

		return mr_world.Container<World::ContainerType::Sphere>().Create(construct);
	}


	void JsonLoader::LoadMaterial(const nlohmann::json& json, Material& material)
	{
		if (json.is_object())
		{
			if (json.contains("file"))
			{
				auto& value = json["file"];
				if (!value.is_string())
					throw Exception("Path to file must be string.");

				mr_world.GetLoader().LoadMTL(static_cast<std::string>(value), material);
				return;
			}

			for (auto& item : json.items())
			{
				auto& key = item.key();
				auto& value = item.value();

				if (key == "name" && value.is_string())
					material.SetName(value);
				else if (key == "color")
					material.SetColor(JsonTo<Graphics::Color>(value));
				else if (key == "metalness" && value.is_number())
					material.SetMetalness(std::clamp(float(value), 0.0f, 1.0f));
				else if (key == "roughness" && value.is_number())
					material.SetRoughness(std::clamp(float(value), 0.0f, 1.0f));
				else if (key == "emission" && value.is_number())
					material.SetEmission(std::clamp(float(value), 0.0f, std::numeric_limits<float>::infinity()));
				else if (key == "ior" && value.is_number())
					material.SetIOR(std::clamp(float(value), 1.0f, std::numeric_limits<float>::infinity()));
				else if (key == "scattering" && value.is_number())
					material.SetScattering(std::clamp(float(value), 0.0f, std::numeric_limits<float>::infinity()));

				else if (key == "texture")
					material.SetTexture(Load<World::ContainerType::Texture>(value));
				else if (key == "normal map")
					material.SetNormalMap(Load<World::ContainerType::NormalMap>(value));
				else if (key == "metalness map")
					material.SetMetalnessMap(Load<World::ContainerType::MetalnessMap>(value));
				else if (key == "roughness map")
					material.SetRoughnessMap(Load<World::ContainerType::RoughnessMap>(value));
				else if (key == "emission map")
					material.SetEmissionMap(Load<World::ContainerType::EmissionMap>(value));
			}
		}
	}

	template <World::ContainerType T>
	void JsonLoader::ObjectLoad(const nlohmann::json& world_json, const std::string& key)
	{
		if (world_json.contains(key))
		{
			auto& json = world_json[key];
			if (json.is_object() || json.is_string())
				Load<T>(json);
			else if (json.is_array())
				for (auto& item : json.items())
					Load<T>(item.value());
		}
	}
	void JsonLoader::LoadWorld(const nlohmann::json& world_json)
	{
		mr_world.DestroyAll();

		if (world_json.contains("Objects"))
		{
			auto& objects_json = world_json["Objects"];

			ObjectLoad<World::ContainerType::Texture>(objects_json, "Texture");
			ObjectLoad<World::ContainerType::NormalMap>(objects_json, "NormalMap");
			ObjectLoad<World::ContainerType::MetalnessMap>(objects_json, "MetalnessMap");
			ObjectLoad<World::ContainerType::RoughnessMap>(objects_json, "RoughnessMap");
			ObjectLoad<World::ContainerType::EmissionMap>(objects_json, "EmissionMap");

			ObjectLoad<World::ContainerType::Material>(objects_json, "Material");
			ObjectLoad<World::ContainerType::MeshStructure>(objects_json, "MeshStructure");

			ObjectLoad<World::ContainerType::Camera>(objects_json, "Camera");

			ObjectLoad<World::ContainerType::PointLight>(objects_json, "PointLight");
			ObjectLoad<World::ContainerType::SpotLight>(objects_json, "SpotLight");
			ObjectLoad<World::ContainerType::DirectLight>(objects_json, "DirectLight");

			ObjectLoad<World::ContainerType::Mesh>(objects_json, "Mesh");
			ObjectLoad<World::ContainerType::Sphere>(objects_json, "Sphere");
		}
		if (world_json.contains("Material"))
		{
			LoadMaterial(world_json["Material"], mr_world.GetMaterial());
		}
		if (world_json.contains("DefaultMaterial"))
		{
			LoadMaterial(world_json["DefaultMaterial"], mr_world.GetDefaultMaterial());
		}
	}
	void JsonLoader::LoadJsonScene(std::ifstream& file, const std::filesystem::path& path)
	{
		m_path = path;
		nlohmann::json scene_json;
		try
		{
			scene_json = nlohmann::json::parse(file, nullptr, true, true);
		}
		catch (nlohmann::json::parse_error& ex)
		{
			throw Exception(
				"Failed to parse file " + path.filename().string() +
				" at path " + path.parent_path().string() +
				" at byte " + std::to_string(ex.byte) + ".\n");
		}

		LoadWorld(scene_json);
	}
}