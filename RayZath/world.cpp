#include "world.h"

namespace RayZath
{
	World::World(size_t maxCamerasCount, size_t maxLightsCount, size_t maxRenderObjectsCount)
		: Updatable(nullptr)
		, m_cameras(this, maxCamerasCount)
		, m_point_lights(this, maxLightsCount)
		, m_spheres(this, maxRenderObjectsCount)
	{}
	World::~World()
	{}

	World::ObjectContainer<Camera>& World::GetCameras()
	{
		return m_cameras;
	}
	const World::ObjectContainer<Camera>& World::GetCameras() const
	{
		return m_cameras;
	}
	
	World::ObjectContainer<PointLight>& World::GetPointLights()
	{
		return m_point_lights;
	}
	const World::ObjectContainer<PointLight>& World::GetPointLights() const
	{
		return m_point_lights;
	}

	World::ObjectContainer<Sphere>& World::GetSpheres()
	{
		return m_spheres;
	}
	const World::ObjectContainer<Sphere>& World::GetSpheres() const
	{
		return m_spheres;
	}

	void World::DestroyAllComponents()
	{
		m_cameras.DestroyAllObjects();

		m_point_lights.DestroyAllObjects();
	}
}