#ifndef WORLD_H
#define WORLD_H

#include "dictionary.hpp"
#include "object_container.hpp"
#include "bvh.hpp"

#include "camera.hpp"
#include "spot_light.hpp"
#include "direct_light.hpp"
#include "instance.hpp"
#include "group.hpp"

#include <tuple>

namespace RayZath::Engine
{
	class Loader;
	class Saver;

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
			Mesh,

			Camera,

			SpotLight,
			DirectLight,

			Instance,

			Group,
		};

		using static_dictionary = Utils::static_dictionary;
		template <ObjectType T>
		using object_t = typename static_dictionary::vt_translate<T>::template with<
			static_dictionary::vt_translation<ObjectType::Texture, Texture>,
			static_dictionary::vt_translation<ObjectType::NormalMap, NormalMap>,
			static_dictionary::vt_translation<ObjectType::MetalnessMap, MetalnessMap>,
			static_dictionary::vt_translation<ObjectType::RoughnessMap, RoughnessMap>,
			static_dictionary::vt_translation<ObjectType::EmissionMap, EmissionMap>,

			static_dictionary::vt_translation<ObjectType::Material, Material>,
			static_dictionary::vt_translation<ObjectType::Mesh, Mesh>,

			static_dictionary::vt_translation<ObjectType::Camera, Camera>,
			static_dictionary::vt_translation<ObjectType::SpotLight, SpotLight>,
			static_dictionary::vt_translation<ObjectType::DirectLight, DirectLight>,

			static_dictionary::vt_translation<ObjectType::Instance, Instance>,
			static_dictionary::vt_translation<ObjectType::Group, Group>>::value;

		template <template <ObjectType... Ts> typename T>
		struct apply_all_types
		{
			using type = T<
				ObjectType::Texture,
				ObjectType::NormalMap,
				ObjectType::MetalnessMap,
				ObjectType::RoughnessMap,
				ObjectType::EmissionMap,
				ObjectType::Material,
				ObjectType::Mesh,
				ObjectType::Camera,
				ObjectType::SpotLight,
				ObjectType::DirectLight,
				ObjectType::Instance,
				ObjectType::Group>;
		};
		
	private:
		template <ObjectType CT>
		static constexpr size_t idx_of = static_dictionary::vv_translate<CT>::template with<
			static_dictionary::vv_translation<ObjectType::Texture, 0>,
			static_dictionary::vv_translation<ObjectType::NormalMap, 1>,
			static_dictionary::vv_translation<ObjectType::MetalnessMap, 2>,
			static_dictionary::vv_translation<ObjectType::RoughnessMap, 3>,
			static_dictionary::vv_translation<ObjectType::EmissionMap, 4>,

			static_dictionary::vv_translation<ObjectType::Material, 5>,
			static_dictionary::vv_translation<ObjectType::Mesh, 6>,

			static_dictionary::vv_translation<ObjectType::Camera, 7>,
			static_dictionary::vv_translation<ObjectType::SpotLight, 8>,
			static_dictionary::vv_translation<ObjectType::DirectLight, 9>,

			static_dictionary::vv_translation<ObjectType::Instance, 10>,
			static_dictionary::vv_translation<ObjectType::Group, 11>>::value;
		template <ObjectType CT>
		static constexpr bool is_subdivided_v = Utils::is::value<CT>::template any_of<ObjectType::Instance>::value;

		std::tuple<
			ObjectContainer<Texture>,
			ObjectContainer<NormalMap>,
			ObjectContainer<MetalnessMap>,
			ObjectContainer<RoughnessMap>,
			ObjectContainer<EmissionMap>,

			ObjectContainer<Material>,
			ObjectContainer<Mesh>,

			ObjectContainer<Camera>,

			ObjectContainer<SpotLight>,
			ObjectContainer<DirectLight>,

			ObjectContainerWithBVH<Instance>,
			ObjectContainer<Group>> m_containers;

		Material m_material;
		Material m_default_material;

		std::unique_ptr<Loader> mp_loader;
		std::unique_ptr<Saver> mp_saver;


	public:
		World(const World& other) = delete;
		World(World&& other) = delete;
		World();


		World& operator=(const World& other) = delete;
		World& operator=(World&& other) = delete;


		template <ObjectType C, std::enable_if_t<!is_subdivided_v<C>, bool> = true>
		auto& container()
		{
			return std::get<idx_of<C>>(m_containers);
		}
		template <ObjectType C, std::enable_if_t<is_subdivided_v<C>, bool> = true>
		auto& container()
		{
			return std::get<idx_of<C>>(m_containers);
		}
		template <ObjectType C, std::enable_if_t<!is_subdivided_v<C>, bool> = true>
		auto& container() const
		{
			return std::get<idx_of<C>>(m_containers);
		}
		template <ObjectType C, std::enable_if_t<is_subdivided_v<C>, bool> = true>
		auto& container() const
		{
			return std::get<idx_of<C>>(m_containers);
		}

		Material& material();
		const Material& material() const;
		Material& defaultMaterial();
		const Material& defaultMaterial() const;

		Loader& loader();
		const Loader& loader() const;
		Saver& saver();
		const Saver& saver() const;

		template <Material::Common M>
		Handle<Material> generateMaterial()
		{
			return container<ObjectType::Material>().create(Material::generateMaterial<M>());
		}

		enum class CommonMesh
		{
			Cube,
			Plane,
			Sphere,
			Cone,
			Cylinder,
			Torus
		};
		template <CommonMesh T>
		struct CommonMeshParameters {};
		template<>
		struct CommonMeshParameters<CommonMesh::Plane>
		{
			uint32_t sides = 4;
			float width = 1.0f, height = 1.0f;

		public:
			CommonMeshParameters(const uint32_t sides = 4, const float width = 1.0f, const float height = 1.0f)
				: sides(sides)
				, width(width)
				, height(height)
			{}
		};
		template<>
		struct CommonMeshParameters<CommonMesh::Sphere>
		{
			uint32_t resolution = 16;
			bool normals = true;
			bool texture_coordinates = true;
			enum class Type
			{
				UVSphere,
				Icosphere
			} type = Type::UVSphere;
		};
		template<>
		struct CommonMeshParameters<CommonMesh::Cone>
		{
			uint32_t side_faces = 16;
			bool normals = true;
			bool texture_coordinates = true;
		};
		template<>
		struct CommonMeshParameters<CommonMesh::Cylinder>
		{
			uint32_t faces = 16;
			bool normals = true;

		public:
			CommonMeshParameters(const uint32_t faces = 16, const bool normals = true)
				: faces(faces)
				, normals(normals)
			{}
		};
		template<>
		struct CommonMeshParameters<CommonMesh::Torus>
		{
			uint32_t minor_resolution = 16, major_resolution = 32;
			float minor_radious = 0.25f, major_radious = 1.0f;
			bool normals = true;
			bool texture_coordinates = true;
		};
		template <CommonMesh T>
		Handle<Mesh> generateMesh(const CommonMeshParameters<T>& parameters);

		void destroyAll();

		void update() override;

		friend class Engine;
	};
}

#endif // !WORLD_H
