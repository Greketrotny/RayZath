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

namespace RayZath
{
	class World : public Updatable
	{
	private:
		struct Containers
			: ObjectContainer<Texture>
			, ObjectContainer<Material>
			, ObjectContainer<MeshStructure>
			, ObjectContainer<Camera>
			, ObjectContainer<PointLight>
			, ObjectContainer<SpotLight>
			, ObjectContainer<DirectLight>
			, ObjectContainerWithBVH<Mesh>
			, ObjectContainerWithBVH<Sphere>
			, ObjectContainer<Plane>
		{
			Containers(
				Updatable* parent,
				uint32_t cameras_capacity,
				uint32_t lights_capacity,
				uint32_t renderables_capacity);
		};
		Containers m_containers;
		Material m_material;
		Material m_default_material;


	private:
		World(
			const uint32_t& maxCamerasCount = 1u, 
			const uint32_t& maxLightsCount = 0x1000u, 
			const uint32_t& maxRenderObjectsCount = 0x1000u);

	
	private:
		template <typename T, typename... Types>
		static constexpr bool is_any_of_v = std::disjunction_v<std::is_same<T, Types>...>;
		template <typename T>
		static constexpr bool IsInLinearStorage =
			is_any_of_v<T,
			Texture, Material, MeshStructure,
			Camera,
			PointLight, SpotLight, DirectLight,
			Plane>;
		template <typename T>
		static constexpr bool IsInSubdividedStorage =
			is_any_of_v<T,
			Mesh, Sphere>;
	public:
		template <typename T, std::enable_if_t<IsInLinearStorage<T>, bool> = true>
		ObjectContainer<T>& Container()
		{
			return static_cast<ObjectContainer<T>&>(m_containers);
		}
		template <typename T, std::enable_if_t<IsInLinearStorage<T>, bool> = true>
		const ObjectContainer<T>& Container() const
		{
			return static_cast<const ObjectContainer<T>&>(m_containers);
		}

		template <typename T, std::enable_if_t<IsInSubdividedStorage<T>, bool> = true>
		ObjectContainerWithBVH<T>& Container()
		{
			return static_cast<ObjectContainerWithBVH<T>&>(m_containers);
		}
		template <typename T, std::enable_if_t<IsInSubdividedStorage<T>, bool> = true>
		const ObjectContainerWithBVH<T>& Container() const
		{
			return static_cast<const ObjectContainerWithBVH<T>&>(m_containers);
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