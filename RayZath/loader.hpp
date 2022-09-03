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
	using key_t = std::string;
	template <World::ObjectType T>
	using map_t = std::unordered_map<key_t, handle_t<T>>;

	template <World::ObjectType... Ts>
	struct LoadedSetView
	{
	public:
		using views_t = std::tuple<std::reference_wrapper<map_t<Ts>>...>;
	private:
		views_t m_map_views;

	public:
		template <World::ObjectType... Us>
		LoadedSetView(views_t views)
			: m_map_views{std::move(views)}
		{}


		template <World::ObjectType T>
		handle_t<T> fetch(const key_t& path) const
		{
			const auto& map = get<T>();
			auto iterator = map.find(path);
			return  (iterator == map.end()) ? handle_t<T>{} : iterator->second;
		}
		template <World::ObjectType T>
		bool add(key_t path, handle_t<T> object)
		{
			auto& map = get<T>();
			const auto [iterator, inserted] = map.insert(std::make_pair(std::move(path), std::move(object)));
			return inserted;
		}
	private:
		template <World::ObjectType T>
		auto& get()
		{
			return std::get<Utils::index_of::value<T>::template in_sequence<Ts...>>(m_map_views).get();
		}
		template <World::ObjectType T>
		const auto& get() const
		{
			return std::get<Utils::index_of::value<T>::template in_sequence<Ts...>>(m_map_views).get();
		}
	};
	template <World::ObjectType... Ts>
	struct LoadedSet
	{
	public:
		using maps_t = std::tuple<map_t<Ts>...>;
	private:
		maps_t m_maps;

	public:
		template <World::ObjectType... Us>
		auto createView()
		{
			return LoadedSetView<Us...>({std::ref(get<Us>())...});
		}
		auto createFullView()
		{
			return createView<Ts...>();
		}
	private:
		template <World::ObjectType T>
		auto& get()
		{
			return std::get<Utils::index_of::value<T>::template in_sequence<Ts...>>(m_maps);
		}
		template <World::ObjectType T>
		const auto& get() const
		{
			return std::get<Utils::index_of::value<T>::template in_sequence<Ts...>>(m_maps);
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
		typename World::object_t<T>::buffer_t LoadMap(const std::string& path);
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

		std::vector<Handle<Material>> LoadMTL(const std::filesystem::path& path);
		std::vector<Handle<Material>> LoadMTL(
			const std::filesystem::path& path,
			loaded_set_view_t loaded_set_view,
			LoadResult& load_result);
		void LoadMTL(const std::filesystem::path& path, Material& material);
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
				Handle<MeshStructure> mesh;
				std::unordered_map<std::string, uint32_t> material_ids;
			};
			std::vector<MeshDesc> meshes;
			std::set<std::filesystem::path> mtllibs;
		};


		OBJLoader(World& world);

	public:
		std::vector<Handle<MeshStructure>> loadMeshes(const std::filesystem::path& file_path);
		std::vector<Handle<Mesh>> loadInstances(const std::filesystem::path& file_path);
		Handle<Group> LoadModel(const std::filesystem::path& file_path);
	private:
		ParseResult parseOBJ(const std::filesystem::path& file_path, LoadResult& load_result);
	};

	class Loader
		: public OBJLoader
	{
	public:
		Loader(World& world);

	public:
		void LoadScene(const std::filesystem::path& path);
	};
}

#endif //!LOADER_H
