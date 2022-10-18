#ifndef LOADER_H
#define LOADER_H

#include "world.hpp"
#include "index_of.hpp"

#include <string>
#include <array>
#include <filesystem>
#include <optional>
#include <memory>
#include <set>

namespace RayZath::Engine
{
	template <World::ObjectType T>
	using handle_t = Handle<World::object_t<T>>;
	using name_key_t = std::string;
	template <World::ObjectType T>
	using name_map_t = std::unordered_map<name_key_t, handle_t<T>>;
	using path_key_t = std::filesystem::path;
	template <World::ObjectType T>
	using path_map_t = std::unordered_map<path_key_t, handle_t<T>>;

	template <World::ObjectType... Ts>
	struct LoadedSetView
	{
	public:
		using name_views_t = std::tuple<std::reference_wrapper<name_map_t<Ts>>...>;
		using path_views_t = std::tuple<std::reference_wrapper<path_map_t<Ts>>...>;
	private:
		name_views_t m_name_views;
		path_views_t m_path_views;

	public:
		template <World::ObjectType... Us>
		LoadedSetView(name_views_t name_views, path_views_t path_views)
			: m_name_views{std::move(name_views)}
			, m_path_views{std::move(path_views)}
		{}


		template <World::ObjectType T>
		auto fetchName(const name_key_t& name) const
		{
			const auto& map = getName<T>();
			auto iterator = map.find(name);
			return  (iterator == map.end()) ? handle_t<T>{} : iterator->second;
		}
		template <World::ObjectType T>
		auto fetchPath(const path_key_t& path) const
		{
			const auto& map = getPath<T>();
			auto iterator = map.find(path);
			return  (iterator == map.end()) ? handle_t<T>{} : iterator->second;
		}
		template <World::ObjectType T>
		bool addName(name_key_t name, handle_t<T> object)
		{
			auto& map = getName<T>();
			const auto [iterator, inserted] = map.insert(std::make_pair(std::move(name), std::move(object)));
			return inserted;
		}
		template <World::ObjectType T>
		bool addPath(path_key_t path, handle_t<T> object)
		{
			auto& map = getPath<T>();
			const auto [iterator, inserted] = map.insert(std::make_pair(std::move(path), std::move(object)));
			return inserted;
		}
	private:
		template <World::ObjectType T>
		auto& getName()
		{
			return std::get<Utils::index_of::value<T>::template in_sequence<Ts...>>(m_name_views).get();
		}
		template <World::ObjectType T>
		const auto& getName() const
		{
			return std::get<Utils::index_of::value<T>::template in_sequence<Ts...>>(m_name_views).get();
		}
		template <World::ObjectType T>
		auto& getPath()
		{
			return std::get<Utils::index_of::value<T>::template in_sequence<Ts...>>(m_path_views).get();
		}
		template <World::ObjectType T>
		const auto& getPath() const
		{
			return std::get<Utils::index_of::value<T>::template in_sequence<Ts...>>(m_path_views).get();
		}
	};
	template <World::ObjectType... Ts>
	struct LoadedSet
	{
	public:
		using name_maps_t = std::tuple<name_map_t<Ts>...>;
		using path_maps_t = std::tuple<path_map_t<Ts>...>;
	private:
		name_maps_t m_name_maps;
		path_maps_t m_path_maps;

	public:
		template <World::ObjectType... Us>
		auto createView()
		{
			return LoadedSetView<Us...>({std::ref(getName<Us>())...}, {std::ref(getPath<Us>())...});
		}
		auto createFullView()
		{
			return createView<Ts...>();
		}
	private:
		template <World::ObjectType T>
		auto& getName()
		{
			return std::get<Utils::index_of::value<T>::template in_sequence<Ts...>>(m_name_maps);
		}
		template <World::ObjectType T>
		const auto& getName() const
		{
			return std::get<Utils::index_of::value<T>::template in_sequence<Ts...>>(m_name_maps);
		}
		template <World::ObjectType T>
		auto& getPath()
		{
			return std::get<Utils::index_of::value<T>::template in_sequence<Ts...>>(m_path_maps);
		}
		template <World::ObjectType T>
		const auto& getPath() const
		{
			return std::get<Utils::index_of::value<T>::template in_sequence<Ts...>>(m_path_maps);
		}
	};

	struct LoadResult
	{
	public:
		enum class MessageType
		{
			Message,
			Warning,
			Error
		};
	private:
		std::vector<std::pair<MessageType, std::string>> m_messages;

	public:
		const auto& messages() { return m_messages; }

		void logMessage(std::string message)
		{
			m_messages.push_back({MessageType::Message, std::move(message)});
		}
		void logWarning(std::string message)
		{
			m_messages.push_back({MessageType::Warning, std::move(message)});
		}
		void logError(std::string message)
		{
			m_messages.push_back({MessageType::Error, std::move(message)});
		}		
	};
	inline std::ostream& operator<<(std::ostream& os, LoadResult& load_result)
	{
		if (load_result.messages().empty())
		{
			os << "load result: empty\n";
			return os;
		}

		os << "load result:\n";
		for (const auto& [type, message] : load_result.messages())
		{
			os << std::setw(10);
			switch (type)
			{
				case LoadResult::MessageType::Message:
					os << "[message] ";
					break;
				case LoadResult::MessageType::Warning:
					os << "[warning] ";
					break;
				case LoadResult::MessageType::Error:
					os << "[error] ";
					break;
				default:
					os << "[unknown] ";
			}
			os << message << '\n';
		}
		return os;
	}

	class LoaderBase
	{
	protected:
		World& mr_world;

	public:
		LoaderBase(World& world);

		std::string_view trimSpaces(const std::string& str);
	};

	class BitmapLoader
		: public LoaderBase
	{
	public:
		BitmapLoader(World& world);

		template <World::ObjectType T>
		typename World::object_t<T>::buffer_t loadMap(const std::string& path);

		std::pair<Texture::buffer_t, EmissionMap::buffer_t> loadHDR(const std::string& path);
	};

	class MTLLoader
		: public BitmapLoader
	{
	public:
		struct MatDesc
		{
			struct MapDesc
			{
				std::filesystem::path path;
				Math::vec2f32 origin;
				Math::vec2f32 scale = Math::vec2f32(1, 1);
			};

			ConStruct<Material> properties;
			std::optional<MapDesc> texture, normal_map, metalness_map, roughness_map, emission_map;
		};
		using loaded_set_view_t = LoadedSetView<
			World::ObjectType::Texture,
			World::ObjectType::NormalMap,
			World::ObjectType::MetalnessMap,
			World::ObjectType::RoughnessMap,
			World::ObjectType::EmissionMap>;

		MTLLoader(World& world);

		std::vector<Handle<Material>> loadMTL(const std::filesystem::path& path);
		std::vector<Handle<Material>> loadMTL(
			const std::filesystem::path& path,
			loaded_set_view_t loaded_set_view,
			LoadResult& load_result);
		void loadMTL(const std::filesystem::path& path, Material& material);
	private:
		std::vector<MatDesc> parseMTL(
			const std::filesystem::path& path,
			LoadResult& load_result);
	};

	class OBJLoader
		: public MTLLoader
	{
	public:
		struct ParseResult
		{
			struct MeshDesc
			{
				Handle<Mesh> mesh;
				std::unordered_map<std::string, uint32_t> material_ids;
			};
			std::vector<MeshDesc> meshes;
			std::set<std::filesystem::path> mtllibs;
		};


		OBJLoader(World& world);

	public:
		std::vector<Handle<Mesh>> loadMeshes(const std::filesystem::path& file_path);
		std::vector<Handle<Instance>> loadInstances(const std::filesystem::path& file_path);
		Handle<Group> loadModel(const std::filesystem::path& file_path);
	private:
		ParseResult parseOBJ(const std::filesystem::path& file_path, LoadResult& load_result);
	};

	class Loader
		: public OBJLoader
	{
	public:
		Loader(World& world);

	public:
		void loadScene(const std::filesystem::path& path);
	};
}

#endif //!LOADER_H
