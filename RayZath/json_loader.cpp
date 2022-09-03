#include "json_loader.hpp"

#include "rzexception.hpp"
#include "loader.hpp"
#include "dictionary.hpp"

#include <variant>

namespace RayZath::Engine
{
	using json_t = JsonLoader::json_t;

	JsonLoader::JsonLoader(std::reference_wrapper<World> world, std::filesystem::path file_path)
		: mr_world(std::move(world))
		, m_path(std::move(file_path))
		, m_loaded_set_view(m_loaded_set.createFullView())
	{}

	std::filesystem::path JsonLoader::makeLoadPath(std::filesystem::path path)
	{
		return makeLoadPath(path, m_path.parent_path());
	}
	std::filesystem::path JsonLoader::makeLoadPath(std::filesystem::path path, std::filesystem::path base)
	{
		if (path.is_absolute())
			return path;
		return base / path;
	}

	template <typename T, std::enable_if_t<std::is_same_v<T, Math::vec3f>, bool> = false>
	T JsonTo(const json_t& vec3_json)
	{
		RZAssert(vec3_json.is_array(), "Value is not an array.");
		RZAssert(vec3_json.size() == 3u, "Array has to have three coordinates.");
		RZAssert(vec3_json[0].is_number() &&
			vec3_json[1].is_number() &&
			vec3_json[2].is_number(),
			"Coordinates should be numbers.");

		auto values = vec3_json.get<std::array<float, 3u>>();
		return T(values[0], values[1], values[2]);
	}
	template <typename T, std::enable_if_t<std::is_same_v<T, Math::vec2f> || std::is_same_v<T, Math::vec2ui32>, bool> = false>
	T JsonTo(const json_t& vec2_json)
	{
		RZAssert(vec2_json.is_array(), "Value is not an array.");
		RZAssert(vec2_json.size() == 2u, "Array has to have two coordinates.");
		RZAssert(vec2_json[0].is_number() &&
			vec2_json[1].is_number(),
			"Coordinates should be numbers.");

		auto values = vec2_json.get<std::array<float, 2u>>();
		using value_t = decltype(std::declval<T>().x);
		return T(value_t(values[0]), value_t(values[1]));
	}
	template <typename T, std::enable_if_t<std::is_same_v<T, Graphics::Color>, bool> = false>
	T JsonTo(const json_t& json)
	{
		RZAssert(json.is_array(), "Value is not an array.");
		RZAssert(json.size() >= 3u, "Color has at least three channels.");

		std::array<uint8_t, 4u> values = {0xF0, 0xF0, 0xF0, 0xFF};
		for (size_t i = 0u; i < json.size(); i++)
		{
			RZAssert(json[i].is_number(), "Color values should be numbers.");
			if (json[i].is_number_float())
				values[i] = uint8_t(std::clamp(float(json[i]), 0.0f, 1.0f) * 255.0f);
			else if (json[i].is_number_integer())
				values[i] = uint8_t(std::clamp<uint32_t>(uint32_t(json[i]), 0u, 255u));
		}

		return T(values[0], values[1], values[2], values[3]);
	}

	template <World::ObjectType T, typename MapT>
	Handle<MapT> JsonLoader::LoadMap(const json_t& json)
	{
		if (json.is_string())
		{	// reference to supposedly alrady loaded map
			auto map_name = static_cast<std::string>(json);
			auto map = m_loaded_set_view.fetch<T>(map_name);
			if (!map)
			{
				m_load_result.logError("\"" + map_name + "\" is not yet a loaded map.");
			}
			return map;
		}
		if (!json.is_object())
		{
			m_load_result.logError("Value of map definition has to be either a string or an object.");
			return {};
		}
		if (!(json.contains("name") && json.contains("file")))
		{
			m_load_result.logError("Map definition has to contain \"name\" and \"file\" properties");
			return {};
		}

		ConStruct<MapT> construct{static_cast<std::string>(json["name"])};
		for (auto& item : json.items())
		{
			auto& key = item.key();
			auto& value = item.value();

			try
			{
				if (key == "filter mode" && value.is_string())
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
					construct.bitmap = mr_world.get().GetLoader().LoadMap<T>(
						makeLoadPath(static_cast<std::string>(value)).string());
			}
			catch (RayZath::Exception& e)
			{
				m_load_result.logError(
					"Failed to load " + key + " property of \"" + construct.name + "\". " +
					e.what());
			}
		}

		auto map = mr_world.get().Container<T>().Create(construct);
		if (!m_loaded_set_view.add<T>(map->GetName(), map))
			m_load_result.logWarning("Loading map with ambigous name \"" + map->GetName() + "\".");
		return map;
	}
	template<> Handle<World::object_t<World::ObjectType::Texture>>
	JsonLoader::Load<World::ObjectType::Texture>(const json_t& json)
	{
		return LoadMap<World::ObjectType::Texture>(json);
	}
	template<> Handle<World::object_t<World::ObjectType::NormalMap>>
	JsonLoader::Load<World::ObjectType::NormalMap>(const json_t& json)
	{
		return LoadMap<World::ObjectType::NormalMap>(json);
	}
	template<> Handle<World::object_t<World::ObjectType::MetalnessMap>>
	JsonLoader::Load<World::ObjectType::MetalnessMap>(const json_t& json)
	{
		return LoadMap<World::ObjectType::MetalnessMap>(json);
	}
	template<> Handle<World::object_t<World::ObjectType::RoughnessMap>>
	JsonLoader::Load<World::ObjectType::RoughnessMap>(const json_t& json)
	{
		return LoadMap<World::ObjectType::RoughnessMap>(json);
	}
	template<> Handle<World::object_t<World::ObjectType::EmissionMap>>
	JsonLoader::Load<World::ObjectType::EmissionMap>(const json_t& json)
	{
		return LoadMap<World::ObjectType::EmissionMap>(json);
	}

	template<> Handle<Material> JsonLoader::Load<World::ObjectType::Material>(const json_t& json)
	{
		if (json.is_string())
		{	// reference to supposedly alrady loaded material
			auto material_name = static_cast<std::string>(json);
			auto material = m_loaded_set_view.fetch<World::ObjectType::Material>(material_name);
			if (!material)
				m_load_result.logError("\"" + material_name + "\" is not yet a loaded material.");
			return material;
		}

		Handle<Material> material = mr_world.get().Container<World::ObjectType::Material>().Create(ConStruct<Material>{});
		LoadMaterial(json, *material);

		if (!m_loaded_set_view.add<World::ObjectType::Material>(material->GetName(), material))
			m_load_result.logWarning("Loading material with ambigous name \"" + material->GetName() + "\".");
		m_load_result.logMessage("Loaded material \"" + material->GetName() + "\".");
		return material;
	}
	void JsonLoader::LoadMaterial(const json_t& json, Material& material)
	{
		if (!json.is_object())
		{
			m_load_result.logError("Value of material definition has to be either a string or an object.");
			return;
		}

		// search for material generation statement and setup material parameters
		generateMaterial(json, material);

		if (json.contains("file"))
		{
			auto& value = json["file"];
			if (!value.is_string())
			{
				m_load_result.logError("Value of \"file\" property must be a string.");
			}
			else
			{
				auto path_str = static_cast<std::string>(value);
				mr_world.get().GetLoader().LoadMTL(makeLoadPath(path_str), material);
			}
		}

		if (json.contains("name") && json["name"].is_string())
			material.SetName(static_cast<std::string>(json["name"]));

		for (auto& item : json.items())
		{
			auto& key = item.key();
			auto& value = item.value();

			try
			{
				if (key == "color")
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
			catch (RayZath::Exception& e)
			{
				m_load_result.logError(
					"Failed to load " + key + " property of \"" + material.GetName() + "\" material. " +
					e.what());
			}
		}
	}
	void JsonLoader::generateMaterial(const json_t& json, Material& material)
	{
		static constexpr std::array generate_statements = {
			"generate gold",
			"generate silver",
			"generate copper",
			"generate glass",
			"generate water",
			"generate mirror",
			"generate rough wood",
			"generate polished wood",
			"generate paper",
			"generate rubber",
			"generate rough plastic",
			"generate polished plastic",
			"generate porcelain"
		};
		const char* generate_statement = nullptr;
		for (const auto& statement : generate_statements)
		{
			if (json.contains(statement))
			{
				generate_statement = statement;
				break;
			}
		}
		if (!generate_statement)
			return;

		ConStruct<Material> construct;
		if (std::strcmp(generate_statement, "generate gold") == 0)
			construct = Material::GenerateMaterial<Material::Common::Gold>();
		else if (std::strcmp(generate_statement, "generate silver") == 0)
			construct = Material::GenerateMaterial<Material::Common::Silver>();
		else if (std::strcmp(generate_statement, "generate copper") == 0)
			construct = Material::GenerateMaterial<Material::Common::Copper>();
		else if (std::strcmp(generate_statement, "generate glass") == 0)
			construct = Material::GenerateMaterial<Material::Common::Glass>();
		else if (std::strcmp(generate_statement, "generate water") == 0)
			construct = Material::GenerateMaterial<Material::Common::Water>();
		else if (std::strcmp(generate_statement, "generate mirror") == 0)
			construct = Material::GenerateMaterial<Material::Common::Mirror>();
		else if (std::strcmp(generate_statement, "generate rough wood") == 0)
			construct = Material::GenerateMaterial<Material::Common::RoughWood>();
		else if (std::strcmp(generate_statement, "generate polished wood") == 0)
			construct = Material::GenerateMaterial<Material::Common::PolishedWood>();
		else if (std::strcmp(generate_statement, "generate paper") == 0)
			construct = Material::GenerateMaterial<Material::Common::Paper>();
		else if (std::strcmp(generate_statement, "generate rubber") == 0)
			construct = Material::GenerateMaterial<Material::Common::Rubber>();
		else if (std::strcmp(generate_statement, "generate rough plastic") == 0)
			construct = Material::GenerateMaterial<Material::Common::RoughPlastic>();
		else if (std::strcmp(generate_statement, "generate polished plastic") == 0)
			construct = Material::GenerateMaterial<Material::Common::PolishedPlastic>();
		else if (std::strcmp(generate_statement, "generate porcelain") == 0)
			construct = Material::GenerateMaterial<Material::Common::Porcelain>();
		else
			return;

		material.SetColor(construct.color);
		material.SetMetalness(construct.metalness);
		material.SetRoughness(construct.roughness);
		material.SetEmission(construct.emission);
		material.SetIOR(construct.ior);
		material.SetScattering(construct.scattering);
	}

	Handle<MeshStructure> JsonLoader::generateMesh(const json_t& json)
	{
		static constexpr std::array generate_statements = {
			"generate cube",
			"generate plane",
			"generate sphere",
			"generate cone",
			"generate cylinder",
			"generate torus"
		};
		const char* generate_statement = nullptr;
		for (const auto& statement : generate_statements)
		{
			if (json.contains(statement))
			{
				auto& generate_json = json[statement];
				if (!generate_json.is_object())
				{
					using namespace std::string_literals;
					m_load_result.logError("value of \""s + statement + "\" generation definition must be an object");
					return {};
				}
				generate_statement = statement;
			}
		}
		if (!generate_statement)
			return {};

		auto& generate_json = json[generate_statement];

		// generate common mesh
		if (std::strcmp(generate_statement, "generate cube") == 0)
		{
			return mr_world.get().GenerateMesh<World::CommonMesh::Cube>(
				World::CommonMeshParameters<World::CommonMesh::Cube>{});
		}
		else if (std::strcmp(generate_statement, "generate plane") == 0)
		{
			World::CommonMeshParameters<World::CommonMesh::Plane> params;
			for (const auto& item : generate_json.items())
			{
				auto& key = item.key();
				auto& value = item.value();

				if (key == "resolution" && value.is_number())
					params.sides = uint32_t(std::clamp(float(value), 3.0f, float(value)));
				if (key == "width" && value.is_number())
					params.width = float(value);
				if (key == "height" && value.is_number())
					params.height = float(value);
			}
			auto mesh = mr_world.get().GenerateMesh<World::CommonMesh::Plane>(params);
			return mesh;
		}
		else if (std::strcmp(generate_statement, "generate sphere") == 0)
		{
			RZAssert(generate_json.is_object(), "mesh generation definition should be object");
			World::CommonMeshParameters<World::CommonMesh::Sphere> params;
			for (const auto& item : generate_json.items())
			{
				auto& key = item.key();
				auto& value = item.value();

				if (key == "resolution" && value.is_number())
					params.resolution = uint32_t(std::clamp(float(value), 3.0f, float(value)));
				if (key == "normals" && value.is_boolean())
					params.normals = bool(value);
				if (key == "texcrds" && value.is_boolean())
					params.texture_coordinates = bool(value);
				if (key == "type" && value.is_string())
				{
					const auto type = static_cast<std::string>(value);
					if (type == "uvsphere")
						params.type = decltype(params)::Type::UVSphere;
					else if (type == "icosphere")
						params.type = decltype(params)::Type::Icosphere;
					else RZThrow("invalid sphere type: " + value);
				}
			}
			return mr_world.get().GenerateMesh<World::CommonMesh::Sphere>(params);
		}
		else if (std::strcmp(generate_statement, "generate cone") == 0)
		{
			RZAssert(generate_json.is_object(), "mesh generation definition should be object");
			World::CommonMeshParameters<World::CommonMesh::Cone> params;
			for (const auto& item : generate_json.items())
			{
				auto& key = item.key();
				auto& value = item.value();

				if (key == "resolution" && value.is_number())
					params.side_faces = uint32_t(std::clamp(float(value), 3.0f, float(value)));
				if (key == "normals" && value.is_boolean())
					params.normals = bool(value);
				if (key == "texcrds" && value.is_boolean())
					params.texture_coordinates = bool(value);
			}
			return mr_world.get().GenerateMesh<World::CommonMesh::Cone>(params);
		}
		else if (std::strcmp(generate_statement, "generate cylinder") == 0)
		{
			RZAssert(generate_json.is_object(), "mesh generation definition should be object");
			World::CommonMeshParameters<World::CommonMesh::Cylinder> params;
			for (const auto& item : generate_json.items())
			{
				auto& key = item.key();
				auto& value = item.value();

				if (key == "resolution" && value.is_number())
					params.faces = uint32_t(std::clamp(float(value), 3.0f, float(value)));
				if (key == "normals" && value.is_boolean())
					params.normals = bool(value);
			}
			return mr_world.get().GenerateMesh<World::CommonMesh::Cylinder>(params);
		}
		else if (std::strcmp(generate_statement, "generate torus") == 0)
		{
			RZAssert(generate_json.is_object(), "mesh generation definition should be object");
			World::CommonMeshParameters<World::CommonMesh::Torus> params;
			for (const auto& item : generate_json.items())
			{
				auto& key = item.key();
				auto& value = item.value();

				if (key == "minor resolution" && value.is_number())
					params.minor_resolution = uint32_t(std::clamp(float(value), 3.0f, float(value)));
				if (key == "major resolution" && value.is_number())
					params.major_resolution = uint32_t(std::clamp(float(value), 3.0f, float(value)));
				if (key == "minor radious" && value.is_number())
					params.minor_radious = std::clamp(float(value), 0.0f, float(value));
				if (key == "major radious" && value.is_number())
					params.major_radious = std::clamp(float(value), 0.0f, float(value));
				if (key == "normals" && value.is_boolean())
					params.normals = bool(value);
				if (key == "texcrds" && value.is_boolean())
					params.texture_coordinates = bool(value);
			}
			return mr_world.get().GenerateMesh<World::CommonMesh::Torus>(params);
		}
		else
		{
			return {};
		}
	}
	template<> Handle<MeshStructure> JsonLoader::Load<World::ObjectType::MeshStructure>(const json_t& json)
	{
		if (json.is_string())
		{	// reference to supposedly alrady loaded mesh
			auto mesh_name = static_cast<std::string>(json);
			auto mesh = m_loaded_set_view.fetch<World::ObjectType::MeshStructure>(mesh_name);
			if (!mesh)
				m_load_result.logError("\"" + mesh_name + "\" is not yet a loaded mesh.");
			return mesh;
		}
		if (!json.is_object())
		{
			m_load_result.logError("Value of mesh definition has to be either a string or an object.");
			return {};
		}
		if (!json.contains("name") && !json.contains("file"))
		{
			m_load_result.logError("mesh definition has to contain \"name\" property, when not loaded from file.");
			return {};
		}

		// find out if this is a definition of common mesh to generate
		std::string mesh_name = json.contains("name") ? static_cast<std::string>(json["name"]) : "default";
		if (auto mesh = generateMesh(json); mesh)
		{
			mesh->SetName(mesh_name);
			m_loaded_set_view.add<World::ObjectType::MeshStructure>(mesh_name, mesh);
			m_load_result.logMessage("Loaded mesh \"" + mesh_name + "\".");
			return mesh;
		}

		if (json.contains("file")) // load mesh from specified .obj file
		{
			const auto& file_name_json = json["file"];
			if (!file_name_json.is_string())
			{
				m_load_result.logError("File name has to be a string.");
			}
			else
			{
				const auto file_name = static_cast<std::string>(file_name_json);
				auto meshes = mr_world.get().GetLoader().loadMeshes(file_name);
				if (meshes.size() != 1)
				{
					m_load_result.logWarning(
						std::to_string(meshes.size()) + " meshes loaded from " +
						file_name + ". Exactly one is expected in scene mesh definition.");
				}
				RZAssert(!meshes.empty(), "no mesh loaded from " + file_name);
				auto& first_mesh = meshes.front();
				m_loaded_set_view.add<World::ObjectType::MeshStructure>(first_mesh->GetName(), first_mesh);
				m_load_result.logMessage("Loaded mesh \"" + first_mesh->GetName() + "\".");
				return first_mesh;
			}
		}

		// assembly mesh from vertices/normals/texcrds
		std::vector<const json_t*> vertices, texcrds, normals, triangles;
		ConStruct<MeshStructure> construct{mesh_name};
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

		Handle<MeshStructure> mesh = mr_world.get().Container<World::ObjectType::MeshStructure>().Create(construct);

		for (auto& vs : vertices)
			for (auto& v : *vs)
				mesh->CreateVertex(JsonTo<Vertex>(v));

		for (auto& ts : texcrds)
			for (auto& t : *ts)
				mesh->CreateTexcrd(JsonTo<Texcrd>(t));

		for (auto& ns : normals)
			for (auto& n : *ns)
				mesh->CreateNormal(JsonTo<Normal>(n));

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

				mesh->CreateTriangle(indices[0], indices[1], indices[2], material_idx);
			}

		if (!m_loaded_set_view.add<World::ObjectType::MeshStructure>(mesh->GetName(), mesh))
			m_load_result.logWarning("Loading mesh with ambigous name \"" + mesh->GetName() + "\".");
		m_load_result.logMessage("Loaded mesh \"" + mesh->GetName() + "\".");
		return mesh;
	}

	template<> Handle<Camera> JsonLoader::Load<World::ObjectType::Camera>(const json_t& json)
	{
		if (!json.is_object())
		{
			m_load_result.logError("Value of camera definition has to be an object.");
			return {};
		}

		ConStruct<Camera> construct;
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

		auto camera = mr_world.get().Container<World::ObjectType::Camera>().Create(construct);
		if (!m_loaded_set_view.add<World::ObjectType::Camera>(camera->GetName(), camera))
			m_load_result.logWarning("Loading camera with ambigous name \"" + camera->GetName() + "\".");
		m_load_result.logMessage("Loaded camera \"" + camera->GetName() + "\".");
		return camera;
	}

	template<> Handle<SpotLight> JsonLoader::Load<World::ObjectType::SpotLight>(const json_t& json)
	{
		if (!json.is_object())
		{
			m_load_result.logError("Value of spot light definition has to be an object.");
			return {};
		}

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

		auto light = mr_world.get().Container<World::ObjectType::SpotLight>().Create(construct);
		if (!m_loaded_set_view.add<World::ObjectType::SpotLight>(light->GetName(), light))
			m_load_result.logWarning("Loading spot light with ambigous name \"" + light->GetName() + "\".");
		m_load_result.logMessage("Loaded spot light \"" + light->GetName() + "\".");
		return light;
	}
	template<> Handle<DirectLight> JsonLoader::Load<World::ObjectType::DirectLight>(const json_t& json)
	{
		if (!json.is_object())
		{
			m_load_result.logError("Value of direct light definition has to be an object.");
			return {};
		}

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

		auto light = mr_world.get().Container<World::ObjectType::DirectLight>().Create(construct);
		if (!m_loaded_set_view.add<World::ObjectType::DirectLight>(light->GetName(), light))
			m_load_result.logWarning("Loading direct light with ambigous name \"" + light->GetName() + "\".");
		m_load_result.logMessage("Loaded direct light \"" + light->GetName() + "\".");
		return light;
	}

	template<> Handle<Mesh> JsonLoader::Load<World::ObjectType::Mesh>(const json_t& json)
	{
		if (!json.is_object())
		{
			m_load_result.logError("Value of instance definition has to be an object.");
			return {};
		}

		Handle<Mesh> instance;
		if (json.contains("file"))
		{
			auto& value = json["file"];
			if (!value.is_string())
			{
				m_load_result.logError("Value of \"file\" property in instance definition must be a string.");
				return {};
			}

			const auto file_name = static_cast<std::string>(value);
			auto instances = mr_world.get().GetLoader().loadInstances(makeLoadPath(file_name));
			if (instances.size() != 1)
			{
				m_load_result.logWarning(
					std::to_string(instances.size()) + " instances loaded from " +
					file_name + ". Exactly one is expected in scene instance definition.");
			}
			if (!instances.empty())
				instance = instances[0];
		}

		if (!instance)
			instance = mr_world.get().Container<World::ObjectType::Mesh>().Create({});

		uint32_t material_count = 0u;
		for (auto& item : json.items())
		{
			auto& key = item.key();
			auto& value = item.value();

			if (key == "name" && value.is_string())
				instance->SetName(static_cast<std::string>(value));
			else if (key == "position")
				instance->SetPosition(JsonTo<Math::vec3f>(value));
			else if (key == "rotation")
				instance->SetRotation(JsonTo<Math::vec3f>(value));
			else if (key == "scale")
				instance->SetScale(JsonTo<Math::vec3f>(value));
			else if (key == "Material")
			{
				if (value.is_object())
				{
					if (material_count < Mesh::GetMaterialCapacity())
						instance->SetMaterial(Load<World::ObjectType::Material, Material>(value), material_count++);
				}
				else if (value.is_array())
				{
					for (auto& m : value)
					{
						if (material_count < Mesh::GetMaterialCapacity())
							instance->SetMaterial(Load<World::ObjectType::Material, Material>(m), material_count++);
					}
				}
				else if (value.is_string())
				{
					const auto material_name = static_cast<std::string>(value);
					if (material_count < Mesh::GetMaterialCapacity())
					{
						auto material = m_loaded_set_view.fetch<World::ObjectType::Material>(material_name);
						if (!material)
						{
							m_load_result.logError(
								"Reference to material \"" + material_name + "\" in the definition of instance " +
								instance->GetName() + " is invalid.");
						}
						else
						{
							instance->SetMaterial(material, material_count++);
						}
					}
				}
			}
			else if (key == "MeshStructure")
			{
				if (instance->GetStructure())
					m_load_result.logWarning(
						"Mesh reference for \"" + instance->GetName() +
						"\" instance already specified. Ignored.");
				else
					instance->SetMeshStructure(Load<World::ObjectType::MeshStructure, MeshStructure>(value));
			}
		}

		if (material_count >= Mesh::GetMaterialCapacity())
		{
			m_load_result.logError(
				"Reached the limit of " + std::to_string(Mesh::GetMaterialCapacity()) +
				" materials per instance in definition of \"" + instance->GetName() + "\".");
		}

		m_load_result.logMessage("Loaded instance \"" + instance->GetName() + "\".");
		if (!m_loaded_set_view.add<World::ObjectType::Mesh>(instance->GetName(), instance))
			m_load_result.logWarning("Loading instance with ambigous name \"" + instance->GetName() + "\".");
		return instance;
	}
	template<> Handle<Group> JsonLoader::Load<World::ObjectType::Group>(const json_t& json)
	{
		if (!json.contains("Group"))
			return {}; // no groups defined in the scene file

		std::unordered_map<
			std::string,
			std::pair<Handle<Group>, std::reference_wrapper<const json_t>>> loaded_groups;

		auto load_group = [this, &loaded_groups](const json_t& group_json) -> void
		{
			if (!group_json.is_object())
			{
				m_load_result.logError("Group definition should be an object.");
				return;
			}

			ConStruct<Group> construct;
			for (auto& item : group_json.items())
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

			if (loaded_groups.find(construct.name) != loaded_groups.end())
			{
				m_load_result.logError("Group with name: " + construct.name + " has already been loaded.");
				return;
			}
			auto group = mr_world.get().Container<World::ObjectType::Group>().Create(construct);
			if (!m_loaded_set_view.add<World::ObjectType::Group>(group->GetName(), group))
				m_load_result.logWarning("Loading group with ambigous name \"" + group->GetName() + "\".");
			loaded_groups.insert({group->GetName(), {group, std::cref(group_json)}});

			if (!group_json.contains("objects"))
				return;

			const auto& objects_json = group_json["objects"];
			if (!objects_json.is_array())
			{
				m_load_result.logError("List of objects in group must be an array.");
				return;
			}
			for (const auto& object_json : objects_json)
			{
				if (!object_json.is_string())
				{
					m_load_result.logError("Object entry in group has to be a string, as a name of previously defined object.");
					continue;
				}

				const auto object_name = static_cast<std::string>(object_json);
				auto object = m_loaded_set_view.fetch<World::ObjectType::Mesh>(object_name);
				if (!object)
				{
					m_load_result.logError(
						"Object \"" + object_name +
						"\" referenced in group \"" + group->GetName() +
						"\" couldn't be found");
					continue;
				}

				Group::link(group, object);
			}
			m_load_result.logMessage("Loaded group \"" + group->GetName() + "\".");
		};
		auto link_groups = [this, &loaded_groups]() -> void
		{
			for (const auto& [group_name, group_reference] : loaded_groups)
			{
				const auto& [group, group_json] = group_reference;
				RZAssertCore(group_json.get().is_object(), "group json expected to be an object");
				if (!group_json.get().contains("groups"))
					continue; // no sub-groups defined in the group

				auto& subgroups_json = group_json.get()["groups"];
				if (!subgroups_json.is_array())
				{
					m_load_result.logError("list of sub-groups in group has to be an array.");
					continue;
				}
				for (const auto& subgroup_json : subgroups_json)
				{
					if (!subgroup_json.is_string())
					{
						m_load_result.logError("Sub-group reference in group has to be a string.");
						continue;
					}
					const auto subgroup_name = static_cast<std::string>(subgroup_json);
					auto group_it = loaded_groups.find(subgroup_name);
					if (group_it == loaded_groups.end())
					{
						m_load_result.logError(
							"Subgroup \"" + subgroup_name +
							"\" referenced in group\"" + group->GetName() +
							"\" couldn't be found.");
						continue;
					}

					const auto& [subgroup, _] = group_it->second;

					// circular group reference detection
					Handle<Group> parent_group = group;
					while (parent_group->group())
					{
						parent_group = parent_group->group();
						if (parent_group == subgroup)
						{
							m_load_result.logError(
								"Circular reference detected in groupping. Group \"" + group->GetName() +
								"\" referencing sub-group \"" + subgroup->GetName() +
								"\" has it as a direct or an indirect parent.");
							break;
						}
					}

					if (!(parent_group == group))
						Group::link(group, subgroup);
				}
			}
		};

		const auto& groups = json["Group"];
		if (groups.is_object())
		{
			load_group(groups);
		}
		else if (groups.is_array())
		{
			// load defined groups
			for (const auto& group : groups)
				load_group(group);

			// link loaded groups between themselves
			link_groups();
		}

		return {};
	}


	template <World::ObjectType T, typename U>
	void JsonLoader::ObjectLoad(const json_t& world_json, const std::string& key)
	{
		auto loadObjects = [&](const auto& object_json)
		{
			try
			{
				Load<T, U>(object_json);
			}
			catch (RayZath::Exception& e)
			{
				m_load_result.logError("Failed to load " + key + ". " + e.what());
			}
		};
		if (world_json.contains(key))
		{
			auto& json = world_json[key];
			if (json.is_array())
			{
				for (const auto& item : json.items())
					loadObjects(item.value());
			}
			else
			{
				loadObjects(json);
			}
		}
	}
	void JsonLoader::LoadWorld(const json_t& world_json)
	{
		mr_world.get().DestroyAll();

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

			ObjectLoad<World::ObjectType::Mesh>(objects_json, "Mesh");
			Load<World::ObjectType::Group>(objects_json);
		}
		if (world_json.contains("Material"))
		{
			LoadMaterial(world_json["Material"], mr_world.get().GetMaterial());
		}
		if (world_json.contains("DefaultMaterial"))
		{
			LoadMaterial(world_json["DefaultMaterial"], mr_world.get().GetDefaultMaterial());
		}
	}
	LoadResult JsonLoader::load()
	{
		// open specified file
		std::ifstream file(m_path, std::ios_base::in);
		RZAssert(file.is_open(), "Failed to open file " + m_path.string());

		try
		{
			LoadWorld(json_t::parse(file, nullptr, true, true));
		}
		catch (json_t::parse_error& ex)
		{
			RZThrow(
				"Failed to parse file " + m_path.filename().string() +
				" at byte " + std::to_string(ex.byte) +
				".\nReason: " + ex.what());
		}

		return m_load_result;
	}
}
