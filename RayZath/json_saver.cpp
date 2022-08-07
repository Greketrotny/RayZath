#include "json_saver.h"

#include "rzexception.h"
#include "saver.h"

#include <variant>

#include <iostream>

namespace RayZath::Engine
{
	using json_t = JsonSaver::json_t;

	template <typename T>
	json_t toJson(const T& value);
	template<>
	json_t toJson<Math::vec3f32>(const Math::vec3f32& value)
	{
		return json_t::array({value.x, value.y, value.z});
	}
	template<>
	json_t toJson<Math::vec2f32>(const Math::vec2f32& value)
	{
		return json_t::array({value.x, value.y});
	}
	template<>
	json_t toJson<Math::vec2u32>(const Math::vec2u32& value)
	{
		return json_t::array({value.x, value.y});
	}
	template<>
	json_t toJson<Graphics::Color>(const Graphics::Color& color)
	{
		return json_t::array({color.red, color.green, color.blue, color.alpha});
	}

	JsonSaver::JsonSaver(World& world)
		: mr_world(world)
	{}

	template<>
	void JsonSaver::save<World::ObjectType::Camera>(json_t& json)
	{
		auto camera_array = json_t::array();
		const auto& cameras = mr_world.Container<World::ObjectType::Camera>();
		for (uint32_t i = 0; i < cameras.GetCount(); i++)
		{
			const auto& camera = cameras[i];
			if (!camera) continue;
			auto unique_name = m_names.uniqueName<World::ObjectType::Camera>(camera->GetName());
			camera_array.push_back({
				{"name", unique_name},
				{"position", toJson(camera->GetPosition())},
				{"rotation", toJson(camera->GetRotation())},
				{"resolution", toJson(camera->GetResolution())},
				{"fov", camera->GetFov().value()},
				{"near plane", camera->GetNearFar().x},
				{"far plane", camera->GetNearFar().y},
				{"focal distance", camera->GetFocalDistance()},
				{"aperture", camera->GetAperture()},
				{"exposure time", camera->GetExposureTime()},
				{"temporal blend", camera->GetTemporalBlend()},
				{"enabled", camera->Enabled()}
				});
			m_names.add<World::ObjectType::Camera>(camera, std::move(unique_name));
		}
		json["Camera"] = std::move(camera_array);
	}
	template<>
	void JsonSaver::save<World::ObjectType::SpotLight>(json_t& json)
	{
		auto light_array = json_t::array();
		const auto& lights = mr_world.Container<World::ObjectType::SpotLight>();
		for (uint32_t i = 0; i < lights.GetCount(); i++)
		{
			const auto& light = lights[i];
			if (!light) continue;
			auto unique_name = m_names.uniqueName<World::ObjectType::SpotLight>(light->GetName());
			light_array.push_back({
				{"name", unique_name},
				{"position", toJson(light->GetPosition())},
				{"direction", toJson(light->GetDirection())},
				{"color", toJson(light->GetColor())},
				{"size", light->GetSize()},
				{"emission", light->GetEmission()},
				{"angle", light->GetBeamAngle()}
				});
			m_names.add<World::ObjectType::SpotLight>(light, std::move(unique_name));
		}
		json["SpotLight"] = std::move(light_array);
	}
	template<>
	void JsonSaver::save<World::ObjectType::DirectLight>(json_t& json)
	{
		auto light_array = json_t::array();
		const auto& lights = mr_world.Container<World::ObjectType::DirectLight>();
		for (uint32_t i = 0; i < lights.GetCount(); i++)
		{
			const auto& light = lights[i];
			if (!light) continue;
			auto unique_name = m_names.uniqueName<World::ObjectType::DirectLight>(light->GetName());
			light_array.push_back({
				{"name", unique_name},
				{"direction", toJson(light->GetDirection())},
				{"color", toJson(light->GetColor())},
				{"emission", light->GetEmission()},
				{"size", light->GetAngularSize()},
				});
			m_names.add<World::ObjectType::DirectLight>(light, std::move(unique_name));
		}
		json["DirectLight"] = std::move(light_array);
	}

	template <World::ObjectType T>
	void JsonSaver::saveMap(const std::string& map_key, json_t& json)
	{
		const auto& maps = mr_world.Container<T>();
		if (maps.GetCount() == 0) return;

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

		auto map_array = json_t::array();
		for (uint32_t i = 0; i < maps.GetCount(); i++)
		{
			const auto& map = maps[i];
			if (!map) continue;
			auto unique_name = m_names.uniqueName<T>(map->GetName());
			map_array.push_back({
				{"name", unique_name},
				{"filter mode", filter_modes.at(map->GetFilterMode())},
				{"address mode", address_modes.at(map->GetAddressMode())},
				{"scale", toJson(map->GetScale())},
				{"rotation", map->GetRotation().value()},
				{"translation", toJson(map->GetTranslation())},
				{"file", mr_world.GetSaver().SaveMap<T>(
					map->GetBitmap(), 
					m_path / Paths::path<T>, 
					unique_name).string()}
				});
			m_names.add<T>(map, std::move(unique_name));
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

	void JsonSaver::saveJsonScene(const Saver::SaveOptions& options)
	{
		RZAssert(options.path.has_filename(), "path must contain file name.json");
		RZAssert(
			options.path.has_extension() && options.path.extension() == ".json", 
			"path must contain file name with .json extension");
		m_path = options.path.parent_path();
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

		// write into file
		std::ofstream file(options.path);
		RZAssert(file.is_open(), "failed to save scene to file " + options.path.string());
		file.exceptions(file.failbit);
		file << m_json.dump(2) << '\n';
	}
}
