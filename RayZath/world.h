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

namespace RayZath
{
	class World : public Updatable
	{
	private:
		ObjectContainer<Material> m_materials;

		ObjectContainer<Camera> m_cameras;

		ObjectContainer<PointLight> m_point_lights;
		ObjectContainer<SpotLight> m_spot_lights;
		ObjectContainer<DirectLight> m_direct_lights;

		ObjectContainerWithBVH<Mesh> m_meshes;
		ObjectContainerWithBVH<Sphere> m_spheres;

		Material m_material;


	private:
		World(
			const uint32_t& maxCamerasCount = 1u, 
			const uint32_t& maxLightsCount = 16u, 
			const uint32_t& maxRenderObjectsCount = 0x1000u);
		~World();


	public:
		ObjectContainer<Material>& GetMaterials();
		const ObjectContainer<Material>& GetMaterials() const;

		ObjectContainer<Camera>& GetCameras();
		const ObjectContainer<Camera>& GetCameras() const;

		ObjectContainer<PointLight>& GetPointLights();
		const ObjectContainer<PointLight>& GetPointLights() const;
		ObjectContainer<SpotLight>& GetSpotLights();
		const ObjectContainer<SpotLight>& GetSpotLights() const;
		ObjectContainer<DirectLight>& GetDirectLights();
		const ObjectContainer<DirectLight>& GetDirectLights() const;

		ObjectContainerWithBVH<Mesh>& GetMeshes();
		const ObjectContainerWithBVH<Mesh>& GetMeshes() const;
		ObjectContainerWithBVH<Sphere>& GetSpheres();
		const ObjectContainerWithBVH<Sphere>& GetSpheres() const;

		Material& GetMaterial();
		const Material& GetMaterial() const;

		void DestroyAllComponents();

		void Update() override;


		friend class Engine;
	};
}

#endif // !WORLD_H