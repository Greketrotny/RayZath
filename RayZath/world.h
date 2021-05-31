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
	class World : public Updatable
	{
	public:
		enum class ContainerType
		{
			Texture,
			NormalMap,
			EmittanceMap,

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
		std::tuple<
			ObjectContainer<Texture>,
			ObjectContainer<NormalMap>,
			ObjectContainer<EmittanceMap>,

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


	private:
		World(const World& other) = delete;
		World(World&& other) = delete;
		World(
			const uint32_t& maxCamerasCount = 1u, 
			const uint32_t& maxLightsCount = 0x1000u, 
			const uint32_t& maxRenderObjectsCount = 0x1000u);


	public:
		World& operator=(const World& other) = delete;
		World& operator=(World&& other) = delete;

	
	private:
		template <ContainerType, ContainerType>
		static constexpr bool is_same_v = false;
		template <ContainerType C>
		static constexpr bool is_same_v<C, C> = true;
		template <ContainerType C1, ContainerType C2>
		struct is_same : std::bool_constant<is_same_v<C1, C2>> {};

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

		void DestroyAll();

		Handle<Material> GenerateGlassMaterial(const Handle<Texture>& texture = {});
		Handle<Material> GenerateMirrorMaterial(const Handle<Texture>& texture = {});
		Handle<Material> GenerateDiffuseMaterial(const Handle<Texture>& texture = {});
		Handle<Material> GenerateGlossyMaterial(const Handle<Texture>& texture = {});

		void Update() override;

		friend class Engine;
	};
}

#endif // !WORLD_H