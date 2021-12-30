#ifndef WORLD_H
#define WORLD_H

#include "object_container.h"
#include "bvh.h"

#include "camera.h"

#include "point_light.h"
#include "spot_light.h"
#include "direct_light.h"

#include "mesh.h"
#include "sphere.h"
#include "plane.h"

#include <tuple>

namespace RayZath
{
	class Loader;

	class World 
		: public Updatable
	{
	public:
		enum class ContainerType
		{
			Texture,
			NormalMap,
			MetalnessMap,
			RoughnessMap,
			EmissionMap,

			Material,
			MeshStructure,

			Camera,

			PointLight,
			SpotLight,
			DirectLight,

			Mesh,
			Sphere,
			Plane
		};
	private:
		template <ContainerType, ContainerType>
		static constexpr bool is_same_v = false;
		template <ContainerType C>
		static constexpr bool is_same_v<C, C> = true;
		template <ContainerType C1, ContainerType C2>
		struct is_same : std::bool_constant<is_same_v<C1, C2>> {};

		template <ContainerType CT, typename T>
		struct translation
		{
			static constexpr auto enum_type = CT;
			using type = T;
		};

		template <bool B, ContainerType CT, typename tr, typename... trs>
		struct reducer;
		template <ContainerType CT, typename tr, typename... trs>
		struct dictionary;

		template <ContainerType CT, typename tr, typename... trs>
		struct reducer<false, CT, tr, trs...>
		{
			using type = typename dictionary<CT, trs...>::type;
		};
		template <ContainerType CT, typename tr, typename... trs>
		struct reducer<true, CT, tr, trs...>
		{
			using type = typename tr::type;
		};
		template <ContainerType CT, typename tr, typename... trs>
		struct dictionary
		{
			using type = typename reducer<is_same_v<CT, tr::enum_type>, CT, tr, trs...>::type;
		};

	public:
		template <ContainerType CT>
		using type_of_t = typename dictionary<CT
			, translation<ContainerType::Texture, Texture>
			, translation<ContainerType::NormalMap, NormalMap>
			, translation<ContainerType::MetalnessMap, MetalnessMap>
			, translation<ContainerType::RoughnessMap, RoughnessMap>
			, translation<ContainerType::EmissionMap, EmissionMap>

			, translation<ContainerType::Material, Material>
			, translation<ContainerType::MeshStructure, MeshStructure>

			, translation<ContainerType::Camera, Camera>

			, translation<ContainerType::PointLight, PointLight>
			, translation<ContainerType::SpotLight, SpotLight>
			, translation<ContainerType::DirectLight, DirectLight>

			, translation<ContainerType::Mesh, Mesh>
			, translation<ContainerType::Sphere, Sphere>
			, translation<ContainerType::Plane, Plane>>::type;

	private:
		std::tuple<
			ObjectContainer<Texture>,
			ObjectContainer<NormalMap>,
			ObjectContainer<MetalnessMap>,
			ObjectContainer<RoughnessMap>,
			ObjectContainer<EmissionMap>,

			ObjectContainer<Material>,
			ObjectContainer<MeshStructure>,

			ObjectContainer<Camera>,

			ObjectContainer<PointLight>,
			ObjectContainer<SpotLight>,
			ObjectContainer<DirectLight>,

			ObjectContainerWithBVH<Mesh>,
			ObjectContainerWithBVH<Sphere>,
			ObjectContainer<Plane>> m_containers;

		Material m_material;
		Material m_default_material;

		std::unique_ptr<Loader> mp_loader;


	private:
		World(const World& other) = delete;
		World(World&& other) = delete;
		World();


	public:
		World& operator=(const World& other) = delete;
		World& operator=(World&& other) = delete;

	
	private:
		template <ContainerType C, ContainerType... Cs>
		static constexpr bool is_any_of_v = std::disjunction_v<is_same<C, Cs>...>;
		template <ContainerType C>
		static constexpr bool is_subdivided_v = is_any_of_v<C,
			ContainerType::Mesh,
			ContainerType::Sphere>;


		template <ContainerType C>
		using container_type = decltype(std::get<size_t(C)>(m_containers));

	public:
		template <ContainerType C, std::enable_if_t<!is_subdivided_v<C>, bool> = true>
		auto& Container()
		{
			return std::get<size_t(C)>(m_containers);
		}
		template <ContainerType C, std::enable_if_t<is_subdivided_v<C>, bool> = true>
		auto& Container()
		{
			return std::get<size_t(C)>(m_containers);
		}
		template <ContainerType C, std::enable_if_t<!is_subdivided_v<C>, bool> = true>
		auto& Container() const
		{
			return std::get<size_t(C)>(m_containers);
		}
		template <ContainerType C, std::enable_if_t<is_subdivided_v<C>, bool> = true>
		auto& Container() const
		{
			return std::get<size_t(C)>(m_containers);
		}


	public:
		Material& GetMaterial();
		const Material& GetMaterial() const;
		Material& GetDefaultMaterial();
		const Material& GetDefaultMaterial() const;

		Loader& GetLoader();
		const Loader& GetLoader() const;

		template <Material::Common M>
		Handle<Material> GenerateMaterial()
		{
			return Container<ContainerType::Material>().Create(Material::GenerateMaterial<M>());
		}

		void DestroyAll();

		void Update() override;

		friend class Engine;
	};
}

#endif // !WORLD_H