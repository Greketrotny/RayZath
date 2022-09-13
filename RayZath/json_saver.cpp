#include "json_saver.hpp"

#include "rzexception.hpp"
#include "saver.hpp"

#include <variant>

#include <iostream>

namespace RayZath::Engine
{
	using json_t = JsonSaver::json_t;

	template <typename T>
	json_t toJson(const T& value);
	template<> json_t toJson<Math::vec3f32>(const Math::vec3f32& value)
	{
		return json_t::array({value.x, value.y, value.z});
	}
	template<> json_t toJson<Math::vec2f32>(const Math::vec2f32& value)
	{
		return json_t::array({value.x, value.y});
	}
	template<> json_t toJson<Math::vec2u32>(const Math::vec2u32& value)
	{
		return json_t::array({value.x, value.y});
	}
	template<> json_t toJson<Graphics::Color>(const Graphics::Color& color)
	{
		return json_t::array({color.red, color.green, color.blue, color.alpha});
	}

	JsonSaver::JsonSaver(World& world)
		: mr_world(world)
	{}

	template<>
	void JsonSaver::save<World::ObjectType::Camera>(json_t& json)
	{
		const auto& cameras = mr_world.container<World::ObjectType::Camera>();
		if (cameras.count() == 0) return;

		auto camera_array = json_t::array();
		for (uint32_t i = 0; i < cameras.count(); i++)
		{
			const auto& camera = cameras[i];
			if (!camera) continue;
			auto unique_name = m_names.uniqueName<World::ObjectType::Camera>(camera->name());
			camera_array.push_back({
				{"name", unique_name},
				{"position", toJson(camera->position())},
				{"rotation", toJson(camera->rotation())},
				{"resolution", toJson(camera->resolution())},
				{"fov", camera->fov().value()},
				{"near plane", camera->nearFar().x},
				{"far plane", camera->nearFar().y},
				{"focal distance", camera->focalDistance()},
				{"aperture", camera->aperture()},
				{"exposure time", camera->exposureTime()},
				{"temporal blend", camera->temporalBlend()},
				{"enabled", camera->enabled()}
				});
			m_names.add<World::ObjectType::Camera>(camera, std::move(unique_name));
		}
		json["Camera"] = std::move(camera_array);
	}
	template<>
	void JsonSaver::save<World::ObjectType::SpotLight>(json_t& json)
	{
		const auto& lights = mr_world.container<World::ObjectType::SpotLight>();
		if (lights.count() == 0) return;

		auto light_array = json_t::array();
		for (uint32_t i = 0; i < lights.count(); i++)
		{
			const auto& light = lights[i];
			if (!light) continue;
			auto unique_name = m_names.uniqueName<World::ObjectType::SpotLight>(light->name());
			light_array.push_back({
				{"name", unique_name},
				{"position", toJson(light->position())},
				{"direction", toJson(light->direction())},
				{"color", toJson(light->color())},
				{"size", light->size()},
				{"emission", light->emission()},
				{"angle", light->GetBeamAngle()}
				});
			m_names.add<World::ObjectType::SpotLight>(light, std::move(unique_name));
		}
		json["SpotLight"] = std::move(light_array);
	}
	template<>
	void JsonSaver::save<World::ObjectType::DirectLight>(json_t& json)
	{
		const auto& lights = mr_world.container<World::ObjectType::DirectLight>();
		if (lights.count() == 0) return;

		auto light_array = json_t::array();
		for (uint32_t i = 0; i < lights.count(); i++)
		{
			const auto& light = lights[i];
			if (!light) continue;
			auto unique_name = m_names.uniqueName<World::ObjectType::DirectLight>(light->name());
			light_array.push_back({
				{"name", unique_name},
				{"direction", toJson(light->direction())},
				{"color", toJson(light->color())},
				{"emission", light->emission()},
				{"size", light->angularSize()},
				});
			m_names.add<World::ObjectType::DirectLight>(light, std::move(unique_name));
		}
		json["DirectLight"] = std::move(light_array);
	}

	template <World::ObjectType T>
	void JsonSaver::saveMap(const std::string& map_key, json_t& json)
	{
		const auto& maps = mr_world.container<T>();
		if (maps.count() == 0) return;

		static const std::unordered_map<typename World::object_t<T>::AddressMode, const char*> address_modes = {
			{World::object_t<T>::AddressMode::Wrap, "wrap"},
			{World::object_t<T>::AddressMode::Clamp, "clamp"},
			{World::object_t<T>::AddressMode::Mirror, "mirror"},
			{World::object_t<T>::AddressMode::Border, "border"}
		};
		static const std::unordered_map<typename World::object_t<T>::FilterMode, const char*> filter_modes = {
			{World::object_t<T>::FilterMode::Point, "point"},
			{World::object_t<T>::FilterMode::Linear, "linear"}
		};

		// save each map
		auto map_array = json_t::array();
		for (uint32_t i = 0; i < maps.count(); i++)
		{
			const auto& map = maps[i];
			if (!map) continue;

			// generate unique name
			auto unique_name = m_names.uniqueName<T>(map->name());
			// save map
			auto path = mr_world.saver().saveMap<T>(map->bitmap(), m_path / Paths::path<T>, unique_name);
			map_array.push_back({
				{"name", unique_name},
				{"filter mode", filter_modes.at(map->filterMode())},
				{"address mode", address_modes.at(map->addressMode())},
				{"scale", toJson(map->scale())},
				{"rotation", map->rotation().value()},
				{"translation", toJson(map->translation())},
				{"file", Saver::relative_path(m_path, path).string()}
				});
			// add saved map object, uniquely generated name and path it has been saved to
			m_names.add<T>(map, std::move(unique_name), std::move(path));
		}
		json[map_key] = std::move(map_array);
	}
	template<> void JsonSaver::save<World::ObjectType::Texture>(json_t& json)
	{
		saveMap<World::ObjectType::Texture>("Texture", json);
	}
	template<> void JsonSaver::save<World::ObjectType::NormalMap>(json_t& json)
	{
		saveMap<World::ObjectType::NormalMap>("NormalMap", json);
	}
	template<> void JsonSaver::save<World::ObjectType::MetalnessMap>(json_t& json)
	{
		saveMap<World::ObjectType::MetalnessMap>("MetalnessMap", json);
	}
	template<> void JsonSaver::save<World::ObjectType::RoughnessMap>(json_t& json)
	{
		saveMap<World::ObjectType::RoughnessMap>("RoughnessMap", json);
	}
	template<> void JsonSaver::save<World::ObjectType::EmissionMap>(json_t& json)
	{
		saveMap<World::ObjectType::EmissionMap>("EmissionMap", json);
	}


	template<>
	void JsonSaver::save<World::ObjectType::Material>(json_t& json)
	{
		const auto& materials = mr_world.container<World::ObjectType::Material>();
		if (materials.count() == 0) return;

		auto material_array = json_t::array();
		for (uint32_t i = 0; i < materials.count(); i++)
		{
			const auto& material = materials[i];
			if (!material) continue;

			MTLSaver::MapsPaths maps_paths;
			if (const auto& texture = material->texture(); texture) maps_paths.texture = MTLSaver::relative_path(
				m_path / Paths::path<World::ObjectType::Material>,
				m_names.path<World::ObjectType::Texture>(material->texture()));
			if (const auto& normal = material->normalMap(); normal) maps_paths.normal = MTLSaver::relative_path(
				m_path / Paths::path<World::ObjectType::Material>,
				m_names.path<World::ObjectType::NormalMap>(material->normalMap()));
			if (const auto& metalness = material->metalnessMap(); metalness) maps_paths.metalness = MTLSaver::relative_path(
				m_path / Paths::path<World::ObjectType::Material>,
				m_names.path<World::ObjectType::MetalnessMap>(material->metalnessMap()));
			if (const auto& roughness = material->roughnessMap(); roughness) maps_paths.roughness = MTLSaver::relative_path(
				m_path / Paths::path<World::ObjectType::Material>,
				m_names.path<World::ObjectType::RoughnessMap>(material->roughnessMap()));
			if (const auto& emission = material->emissionMap(); emission) maps_paths.emission = MTLSaver::relative_path(
				m_path / Paths::path<World::ObjectType::Material>,
				m_names.path<World::ObjectType::EmissionMap>(material->emissionMap()));

			// generate unique name
			auto unique_name = m_names.uniqueName<World::ObjectType::Material>(material->name());
			// save material
			auto path = mr_world.saver().saveMTL(
				*material,
				m_path / Paths::path<World::ObjectType::Material>,
				unique_name,
				maps_paths);
			material_array.push_back({
				{"name", unique_name},
				{"file", Saver::relative_path(m_path, path).string()}});
			// add saved map object, uniquely generated name and path it has been saved to
			m_names.add<World::ObjectType::Material>(material, std::move(unique_name), std::move(path));
		}
		json["Material"] = std::move(material_array);
	}
	template<>
	void JsonSaver::save<World::ObjectType::MeshStructure>(json_t& json)
	{
		const auto& meshes = mr_world.container<World::ObjectType::MeshStructure>();
		if (meshes.count() == 0) return;

		auto mesh_array = json_t::array();
		for (uint32_t i = 0; i < meshes.count(); i++)
		{
			const auto& mesh = meshes[i];
			if (!mesh) continue;

			// generate unique name
			auto unique_name = m_names.uniqueName<World::ObjectType::MeshStructure>(mesh->name());

			auto path = mr_world.saver().saveOBJ(
				*mesh, m_path / Paths::path<World::ObjectType::MeshStructure> / (unique_name + ".obj"),
				std::nullopt,
				{});

			mesh_array.push_back({
				{"name", unique_name},
				{"file", Saver::relative_path(m_path, path).string()}});
			// add saved map object, uniquely generated name and path it has been saved to
			m_names.add<World::ObjectType::MeshStructure>(mesh, std::move(unique_name), std::move(path));
		}
		json["MeshStructure"] = std::move(mesh_array);
	}

	template <>
	void JsonSaver::save<World::ObjectType::Mesh>(json_t& json)
	{
		const auto& instances = mr_world.container<World::ObjectType::Mesh>();
		if (instances.count() == 0) return;

		auto instance_array = json_t::array();
		for (uint32_t instance_idx = 0; instance_idx < instances.count(); instance_idx++)
		{
			const auto& instance = instances[instance_idx];
			if (!instance) continue;

			// generate unique name
			auto unique_name = m_names.uniqueName<World::ObjectType::Mesh>(instance->name());

			// write instance properties
			json_t instance_json = {
				{"name", unique_name},
				{"position", toJson(instance->transformation().position())},
				{"rotation", toJson(instance->transformation().rotation())},
				{"scale", toJson(instance->transformation().scale())}};

			// write all materials
			json_t materials_json;
			for (uint32_t mat_idx = 0; mat_idx < instance->materialCapacity(); mat_idx++)
			{
				const auto& material = instance->material(mat_idx);
				if (!material) continue;

				if (materials_json.empty())
					materials_json = json_t::array();

				// save material as a string - name previously saved during saving materials
				materials_json.push_back(std::string(m_names.name<World::ObjectType::Material>(material)));
			}
			if (!materials_json.empty())
				instance_json["Material"] = std::move(materials_json);

			// write mesh
			if (const auto& mesh = instance->meshStructure())
			{
				instance_json["MeshStructure"] = std::string(m_names.name<World::ObjectType::MeshStructure>(mesh));
			}
			instance_array.push_back(std::move(instance_json));

			// add saved instance with uniquely generated name
			m_names.add<World::ObjectType::Mesh>(instance, std::move(unique_name));
		}
		json["Mesh"] = std::move(instance_array);
	}
	template<>
	void JsonSaver::save<World::ObjectType::Group>(json_t& json)
	{
		const auto& groups = mr_world.container<World::ObjectType::Group>();
		if (groups.count() == 0) return;

		// register all groups with generated unique names
		for (uint32_t i = 0; i < groups.count(); i++)
		{
			const auto& group = groups[i];
			if (!group) continue;

			// generate and add unique group name
			m_names.add<World::ObjectType::Group>(
				group,
				m_names.uniqueName<World::ObjectType::Group>(group->name()));
		}

		// write groups to json
		auto group_array = json_t::array();
		for (uint32_t i = 0; i < groups.count(); i++)
		{
			const auto& group = groups[i];
			if (!group) continue;

			// write group properties
			json_t group_json = {
				{"name", m_names.name<World::ObjectType::Group>(group)},
				{"position", toJson(group->transformation().position())},
				{"rotation", toJson(group->transformation().rotation())},
				{"scale", toJson(group->transformation().scale())}};

			// write groupped instances
			if (!group->objects().empty())
			{
				auto instances_json = json_t::array();
				for (const auto& instance : group->objects())
				{
					if (!instance) continue;
					instances_json.push_back(std::string(m_names.name<World::ObjectType::Mesh>(instance)));
				}
				group_json["objects"] = std::move(instances_json);
			}

			// write subgroups
			if (!group->groups().empty())
			{
				auto subgroups_json = json_t::array();
				for (const auto& subgroup : group->groups())
				{
					if (!subgroup) continue;
					subgroups_json.push_back(std::string(m_names.name<World::ObjectType::Group>(subgroup)));
				}
				group_json["groups"] = std::move(subgroups_json);
			}
			group_array.push_back(std::move(group_json));
		}

		json["Group"] = std::move(group_array);
	}

	void JsonSaver::saveSpecialMaterial(const char* key, const Material& material, json_t& json)
	{
		MTLSaver::MapsPaths maps_paths;
		if (const auto& texture = material.texture(); texture) maps_paths.texture = MTLSaver::relative_path(
			m_path / Paths::path<World::ObjectType::Material>,
			m_names.path<World::ObjectType::Texture>(material.texture()));
		if (const auto& normal = material.normalMap(); normal) maps_paths.normal = MTLSaver::relative_path(
			m_path / Paths::path<World::ObjectType::Material>,
			m_names.path<World::ObjectType::NormalMap>(material.normalMap()));
		if (const auto& metalness = material.metalnessMap(); metalness) maps_paths.metalness = MTLSaver::relative_path(
			m_path / Paths::path<World::ObjectType::Material>,
			m_names.path<World::ObjectType::MetalnessMap>(material.metalnessMap()));
		if (const auto& roughness = material.roughnessMap(); roughness) maps_paths.roughness = MTLSaver::relative_path(
			m_path / Paths::path<World::ObjectType::Material>,
			m_names.path<World::ObjectType::RoughnessMap>(material.roughnessMap()));
		if (const auto& emission = material.emissionMap(); emission) maps_paths.emission = MTLSaver::relative_path(
			m_path / Paths::path<World::ObjectType::Material>,
			m_names.path<World::ObjectType::EmissionMap>(material.emissionMap()));

		// save material
		auto path = mr_world.saver().saveMTL(
			material,
			m_path / Paths::path<World::ObjectType::Material>,
			key,
			maps_paths);

		json[key] = {
			{"name", key },
			{"file", Saver::relative_path(m_path, path).string()}};
	}

	void JsonSaver::saveJsonScene(const Saver::SaveOptions& options)
	{
		RZAssert(options.path.has_filename(), "path must contain file name.json");
		RZAssert(
			options.path.has_extension() && options.path.extension() == ".json",
			"path must contain file name with .json extension");
		m_path = options.path;
		m_path.remove_filename();
		RZAssert(std::filesystem::is_empty(m_path), "specified folder must be empty");

		m_json = {};
		m_names.reset();

		auto& objects_json = m_json["Objects"] = {};
		save<World::ObjectType::Camera>(objects_json);
		save<World::ObjectType::SpotLight>(objects_json);
		save<World::ObjectType::DirectLight>(objects_json);

		save<World::ObjectType::Texture>(objects_json);
		save<World::ObjectType::NormalMap>(objects_json);
		save<World::ObjectType::MetalnessMap>(objects_json);
		save<World::ObjectType::RoughnessMap>(objects_json);
		save<World::ObjectType::EmissionMap>(objects_json);

		save<World::ObjectType::Material>(objects_json);
		save<World::ObjectType::MeshStructure>(objects_json);

		save<World::ObjectType::Mesh>(objects_json);
		save<World::ObjectType::Group>(objects_json);

		saveSpecialMaterial("Material", mr_world.material(), m_json);
		saveSpecialMaterial("DefaultMaterial", mr_world.defaultMaterial(), m_json);

		// write into file
		std::ofstream file(options.path);
		RZAssert(file.is_open(), "failed to save scene to file " + options.path.string());
		file.exceptions(file.failbit);
		file << m_json.dump(2) << '\n';
	}
}
