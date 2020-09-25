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
		ObjectContainer<Camera> m_cameras;

		ObjectContainer<PointLight> m_point_lights;
		ObjectContainer<SpotLight> m_spot_lights;
		ObjectContainer<DirectLight> m_direct_lights;

		ObjectContainerWithBVH<Mesh> m_meshes;
		ObjectContainerWithBVH<Sphere> m_spheres;


	private:
		World(
			size_t maxCamerasCount = 16u, 
			size_t maxLightsCount = 16u, 
			size_t maxRenderObjectsCount = 1024u);
		~World();


	public:
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


		void DestroyAllComponents();


		friend class Engine;
	};
}

#endif // !WORLD_H