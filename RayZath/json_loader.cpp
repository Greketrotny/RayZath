#include "json_loader.h"

#include "loader.h"

namespace RayZath::Engine
{
	using json_t = JsonLoader::json_t;

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

	template <World::ObjectType T, typename MapT>
	Handle<MapT> JsonLoader::LoadMap(const nlohmann::json& json)
	{
		if (json.is_string())
			return mr_world.Container<T>()[static_cast<std::string>(json)];
		if (!json.is_object())
			return {};

		ConStruct<MapT> construct;
		for (auto& item : json.items())
		{
			auto& key = item.key();
			auto& value = item.value();

			if (key == "name" && value.is_string())
				construct.name = value;
			else if (key == "filter mode" && value.is_string())
			{
				if (value == "point") construct.filter_mode = MapT::FilterMode::Point;
				else if (value == "linear") construct.filter_mode = MapT::FilterMode::Linear;
			}
			else if (key == "address mode" && value.is_string())
			{
				if (value == "wrap") construct.address_mode = MapT::AddressMode::Wrap;
				else if (value == "clamp") construct.address_mode = MapT::AddressMode::Clamp;
				else if (value == "mirror") construct.address_mode = MapT::AddressMode::Mirror;
				else if (value == "border") construct.address_mode = MapT::AddressMode::Border;
			}
			else if (key == "scale" && value.is_array())
				construct.scale = JsonTo<Math::vec2f>(value);
			else if (key == "rotation" && value.is_number())
				construct.rotation = value;
			else if (key == "translation" && value.is_array())
				construct.translation = JsonTo<Math::vec2f>(value);
			else if (key == "file" && value.is_string())
				construct.bitmap = mr_world.GetLoader().LoadMap<T>(
					ModifyPath(static_cast<std::string>(value)).string());
		}

		return mr_world.Container<T>().Create(construct);
	}
	template<> 
	Handle<World::object_t<World::ObjectType::Texture>> JsonLoader::Load<World::ObjectType::Texture>(const nlohmann::json& json)
	{
		return LoadMap<World::ObjectType::Texture>(json);
	}
	template<>
	Handle<World::object_t<World::ObjectType::NormalMap>> JsonLoader::Load<World::ObjectType::NormalMap>(const nlohmann::json& json)
	{
		return LoadMap<World::ObjectType::NormalMap>(json);
	}
	template<>
	Handle<World::object_t<World::ObjectType::MetalnessMap>> JsonLoader::Load<World::ObjectType::MetalnessMap>(const nlohmann::json& json)
	{
		return LoadMap<World::ObjectType::MetalnessMap>(json);
	}
	template<>
	Handle<World::object_t<World::ObjectType::RoughnessMap>> JsonLoader::Load<World::ObjectType::RoughnessMap>(const nlohmann::json& json)
	{
		return LoadMap<World::ObjectType::RoughnessMap>(json);
	}
	template<>
	Handle<World::object_t<World::ObjectType::EmissionMap>> JsonLoader::Load<World::ObjectType::EmissionMap>(const nlohmann::json& json)
	{
		return LoadMap<World::ObjectType::EmissionMap>(json);
	}

	template<> Handle<Material> JsonLoader::Load<World::ObjectType::Material>(const nlohmann::json& json)
	{
		if (json.is_string())
		{
			return mr_world.Container<World::ObjectType::Material>()
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
					construct.texture = Load<World::ObjectType::Texture>(value);
				else if (key == "normal map")
					construct.normal_map = Load<World::ObjectType::NormalMap>(value);
				else if (key == "metalness map")
					construct.metalness_map = Load<World::ObjectType::MetalnessMap>(value);
				else if (key == "roughness map")
					construct.roughness_map = Load<World::ObjectType::RoughnessMap>(value);
				else if (key == "emission map")
					construct.emission_map = Load<World::ObjectType::EmissionMap>(value);
			}

			return mr_world.Container<World::ObjectType::Material>().Create(construct);
		}

		return {};
	}
	template<> Handle<MeshStructure> JsonLoader::Load<World::ObjectType::MeshStructure>(const nlohmann::json& json)
	{
		if (json.is_string())
		{
			return mr_world.Container<World::ObjectType::MeshStructure>()
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
				mr_world.Container<World::ObjectType::MeshStructure>().Create(construct);
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

	template<> Handle<Camera> JsonLoader::Load<World::ObjectType::Camera>(const nlohmann::json& camera_json)
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

		return mr_world.Container<World::ObjectType::Camera>().Create(construct);
	}

	template<> Handle<SpotLight> JsonLoader::Load<World::ObjectType::SpotLight>(const nlohmann::json& json)
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
		}

		return mr_world.Container<World::ObjectType::SpotLight>().Create(construct);
	}
	template<> Handle<DirectLight> JsonLoader::Load<World::ObjectType::DirectLight>(const nlohmann::json& json)
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

		return mr_world.Container<World::ObjectType::DirectLight>().Create(construct);
	}

	template<> Handle<Group> JsonLoader::Load<World::ObjectType::Mesh>(const nlohmann::json& json)
	{
		if (!json.is_object())
			return {};

		if (json.contains("file"))
		{
			auto& value = json["file"];
			if (!value.is_string())
				throw Exception("Path to .obj. file should be string.");

			auto group = mr_world.GetLoader().LoadOBJ(ModifyPath(static_cast<std::string>(value)));

			for (auto& item : json.items())
			{
				auto& key = item.key();
				auto& value = item.value();

				if (key == "name" && value.is_string())
					group->SetName(value);
				else if (key == "position")
					group->transformation().SetPosition(JsonTo<Math::vec3f>(value));
				else if (key == "rotation")
					group->transformation().SetRotation(JsonTo<Math::vec3f>(value));
				else if (key == "scale")
					group->transformation().SetScale(JsonTo<Math::vec3f>(value));
			}

			return group;
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
			else if (key == "scale")
				construct.scale = JsonTo<Math::vec3f>(value);
			else if (key == "Material")
			{
				if (value.is_object())
				{
					if (material_count < Mesh::GetMaterialCapacity())
						construct.material[material_count++] =
						Load<World::ObjectType::Material, Material>(value);
				}
				else if (value.is_array())
				{
					for (auto& m : value)
					{
						if (material_count < Mesh::GetMaterialCapacity())
							construct.material[material_count++] =
							Load<World::ObjectType::Material, Material>(m);
					}
				}
				else if (value.is_string())
				{
					if (material_count < Mesh::GetMaterialCapacity())
						construct.material[material_count++] =
						mr_world.Container<World::ObjectType::Material>()
						[static_cast<std::string>(value)];
				}
			}
			else if (key == "MeshStructure")
			{
				if (construct.mesh_structure)
					throw Exception("Mesh structure already defined.");

				construct.mesh_structure =
					Load<World::ObjectType::MeshStructure, MeshStructure>(value);
			}
		}

		mr_world.Container<World::ObjectType::Mesh>().Create(construct);
		return {};
	}
	template<> Handle<Group> JsonLoader::Load<World::ObjectType::Group>(const nlohmann::json& json)
	{
		if (!json.is_object())
			return {};

		ConStruct<Group> construct;

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
			else if (key == "scale")
				construct.scale = JsonTo<Math::vec3f>(value);
		}

		auto group = mr_world.Container<World::ObjectType::Group>().Create(construct);
		RZAssert(bool(group), "group was null");

		for (auto& item : json.items())
		{
			auto& key = item.key();
			auto& value = item.value();

			if (key == "objects" && value.is_array())
			{
				for (const auto& object_name : value)
				{
					if (!object_name.is_string()) continue;

					auto object = mr_world.Container<World::ObjectType::Mesh>()[static_cast<std::string>(object_name)];
					if (!object) continue;

					Group::link(group, object);
				}
			}
			else if (key == "groups")
			{
				for (const auto& group_name : value)
				{
					if (!group_name.is_string()) continue;

					auto subgroup = mr_world.Container<World::ObjectType::Mesh>()[static_cast<std::string>(group_name)];
					if (!subgroup) continue;

					Group::link(group, subgroup);
				}
			}
		}

		return group;
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
					material.SetTexture(Load<World::ObjectType::Texture, Texture>(value));
				else if (key == "normal map")
					material.SetNormalMap(Load<World::ObjectType::NormalMap, NormalMap>(value));
				else if (key == "metalness map")
					material.SetMetalnessMap(Load<World::ObjectType::MetalnessMap, MetalnessMap>(value));
				else if (key == "roughness map")
					material.SetRoughnessMap(Load<World::ObjectType::RoughnessMap, RoughnessMap>(value));
				else if (key == "emission map")
					material.SetEmissionMap(Load<World::ObjectType::EmissionMap, EmissionMap>(value));
			}
		}
	}

	template <World::ObjectType T, typename U>
	void JsonLoader::ObjectLoad(const nlohmann::json& world_json, const std::string& key)
	{
		if (world_json.contains(key))
		{
			auto& json = world_json[key];
			if (json.is_object() || json.is_string())
				Load<T, U>(json);
			else if (json.is_array())
				for (auto& item : json.items())
					Load<T, U>(item.value());
		}
	}
	void JsonLoader::LoadWorld(const nlohmann::json& world_json)
	{
		mr_world.DestroyAll();

		if (world_json.contains("Objects"))
		{
			auto& objects_json = world_json["Objects"];

			ObjectLoad<World::ObjectType::Texture>(objects_json, "Texture");
			ObjectLoad<World::ObjectType::NormalMap>(objects_json, "NormalMap");
			ObjectLoad<World::ObjectType::MetalnessMap>(objects_json, "MetalnessMap");
			ObjectLoad<World::ObjectType::RoughnessMap>(objects_json, "RoughnessMap");
			ObjectLoad<World::ObjectType::EmissionMap>(objects_json, "EmissionMap");

			ObjectLoad<World::ObjectType::Material>(objects_json, "Material");
			ObjectLoad<World::ObjectType::MeshStructure>(objects_json, "MeshStructure");

			ObjectLoad<World::ObjectType::Camera>(objects_json, "Camera");

			ObjectLoad<World::ObjectType::SpotLight>(objects_json, "SpotLight");
			ObjectLoad<World::ObjectType::DirectLight>(objects_json, "DirectLight");

			ObjectLoad<World::ObjectType::Mesh, Group>(objects_json, "Mesh");
			ObjectLoad<World::ObjectType::Group>(objects_json, "Group");
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