#ifndef WORLD_H
#define WORLD_H

#include "static_dictionary.h"
#include "object_container.h"
#include "bvh.h"

#include "camera.h"
#include "spot_light.h"
#include "direct_light.h"
#include "mesh.h"
#include "group.h"

#include <tuple>

namespace RayZath::Engine
{
	class Loader;

	class World
		: public Updatable
	{
	public:
		enum class ObjectType
		{
			Texture,
			NormalMap,
			MetalnessMap,
			RoughnessMap,
			EmissionMap,

			Material,
			MeshStructure,

			Camera,

			SpotLight,
			DirectLight,

			Mesh,

			Group,
		};

		using static_dictionary = RayZath::Utils::static_dictionary;
		template <ObjectType T>
		using object_t = static_dictionary::vt_translate<T>::template with<
			static_dictionary::vt_translation<ObjectType::Texture, Texture>,
			static_dictionary::vt_translation<ObjectType::NormalMap, NormalMap>,
			static_dictionary::vt_translation<ObjectType::MetalnessMap, MetalnessMap>,
			static_dictionary::vt_translation<ObjectType::RoughnessMap, RoughnessMap>,
			static_dictionary::vt_translation<ObjectType::EmissionMap, EmissionMap>,

			static_dictionary::vt_translation<ObjectType::Material, Material>,
			static_dictionary::vt_translation<ObjectType::MeshStructure, MeshStructure>,

			static_dictionary::vt_translation<ObjectType::Camera, Camera>,
			static_dictionary::vt_translation<ObjectType::SpotLight, SpotLight>,
			static_dictionary::vt_translation<ObjectType::DirectLight, DirectLight>,

			static_dictionary::vt_translation<ObjectType::Mesh, Mesh>,
			static_dictionary::vt_translation<ObjectType::Group, Group>>::template value;
		
	private:
		template <ObjectType CT>
		static constexpr size_t idx_of = static_dictionary::vv_translate<CT>::template with<
			static_dictionary::vv_translation<ObjectType::Texture, 0>,
			static_dictionary::vv_translation<ObjectType::NormalMap, 1>,
			static_dictionary::vv_translation<ObjectType::MetalnessMap, 2>,
			static_dictionary::vv_translation<ObjectType::RoughnessMap, 3>,
			static_dictionary::vv_translation<ObjectType::EmissionMap, 4>,

			static_dictionary::vv_translation<ObjectType::Material, 5>,
			static_dictionary::vv_translation<ObjectType::MeshStructure, 6>,

			static_dictionary::vv_translation<ObjectType::Camera, 7>,
			static_dictionary::vv_translation<ObjectType::SpotLight, 8>,
			static_dictionary::vv_translation<ObjectType::DirectLight, 9>,

			static_dictionary::vv_translation<ObjectType::Mesh, 10>,
			static_dictionary::vv_translation<ObjectType::Group, 11>>::value;
		template <ObjectType CT>
		static constexpr bool is_subdivided_v = RayZath::Utils::is::value<CT>::template any_of<ObjectType::Mesh>::value;

		std::tuple<
			ObjectContainer<Texture>,
			ObjectContainer<NormalMap>,
			ObjectContainer<MetalnessMap>,
			ObjectContainer<RoughnessMap>,
			ObjectContainer<EmissionMap>,

			ObjectContainer<Material>,
			ObjectContainer<MeshStructure>,

			ObjectContainer<Camera>,

			ObjectContainer<SpotLight>,
			ObjectContainer<DirectLight>,

			ObjectContainerWithBVH<Mesh>,
			ObjectContainer<Group>> m_containers;

		Material m_material;
		Material m_default_material;

		std::unique_ptr<Loader> mp_loader;


	public:
		World(const World& other) = delete;
		World(World&& other) = delete;
		World();


		World& operator=(const World& other) = delete;
		World& operator=(World&& other) = delete;


		template <ObjectType C, std::enable_if_t<!is_subdivided_v<C>, bool> = true>
		auto& Container()
		{
			return std::get<idx_of<C>>(m_containers);
		}
		template <ObjectType C, std::enable_if_t<is_subdivided_v<C>, bool> = true>
		auto& Container()
		{
			return std::get<idx_of<C>>(m_containers);
		}
		template <ObjectType C, std::enable_if_t<!is_subdivided_v<C>, bool> = true>
		auto& Container() const
		{
			return std::get<idx_of<C>>(m_containers);
		}
		template <ObjectType C, std::enable_if_t<is_subdivided_v<C>, bool> = true>
		auto& Container() const
		{
			return std::get<idx_of<C>>(m_containers);
		}

		Material& GetMaterial();
		const Material& GetMaterial() const;
		Material& GetDefaultMaterial();
		const Material& GetDefaultMaterial() const;

		Loader& GetLoader();
		const Loader& GetLoader() const;

		template <Material::Common M>
		Handle<Material> GenerateMaterial()
		{
			return Container<ObjectType::Material>().Create(Material::GenerateMaterial<M>());
		}

		void DestroyAll();

		void Update() override;

		friend class Engine;
	};
}

#endif // !WORLD_H