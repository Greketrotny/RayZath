#include "world.h"

namespace RayZath
{
	// ~~~~~~~~ [CLASS] World ~~~~~~~~
	World::World(
		const uint32_t& maxCamerasCount, 
		const uint32_t& maxLightsCount, 
		const uint32_t& maxRenderObjectsCount)
		: Updatable(nullptr)
		, m_cameras(this, maxCamerasCount)
		, m_point_lights(this, maxLightsCount)
		, m_spot_lights(this, maxLightsCount)
		, m_direct_lights(this, maxLightsCount)
		, m_meshes(this, maxRenderObjectsCount)
		, m_spheres(this, maxRenderObjectsCount)
		, m_material(
			Graphics::Color(0x10, 0x10, 0x10, 0xFF), 
			0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 0.0f)
	{}
	World::~World()
	{}

	ObjectContainer<Camera>& World::GetCameras()
	{
		return m_cameras;
	}
	const ObjectContainer<Camera>& World::GetCameras() const
	{
		return m_cameras;
	}
	
	ObjectContainer<PointLight>& World::GetPointLights()
	{
		return m_point_lights;
	}
	const ObjectContainer<PointLight>& World::GetPointLights() const
	{
		return m_point_lights;
	}
	ObjectContainer<SpotLight>& World::GetSpotLights()
	{
		return m_spot_lights;
	}
	const ObjectContainer<SpotLight>& World::GetSpotLights() const
	{
		return m_spot_lights;
	}
	ObjectContainer<DirectLight>& World::GetDirectLights()
	{
		return m_direct_lights;
	}
	const ObjectContainer<DirectLight>& World::GetDirectLights() const
	{
		return m_direct_lights;
	}

	ObjectContainerWithBVH<Mesh>& World::GetMeshes()
	{
		return m_meshes;
	}
	const ObjectContainerWithBVH<Mesh>& World::GetMeshes() const
	{
		return m_meshes;
	}
	ObjectContainerWithBVH<Sphere>& World::GetSpheres()
	{
		return m_spheres;
	}
	const ObjectContainerWithBVH<Sphere>& World::GetSpheres() const
	{
		return m_spheres;
	}

	Material& World::GetMaterial()
	{
		return m_material;
	}
	const Material& World::GetMaterial() const
	{
		return m_material;
	}

	void World::DestroyAllComponents()
	{
		m_cameras.DestroyAllObjects();

		m_point_lights.DestroyAllObjects();
		m_spot_lights.DestroyAllObjects();
		m_direct_lights.DestroyAllObjects();

		m_meshes.DestroyAllObjects();
		m_spheres.DestroyAllObjects();
	}

	void World::Update()
	{
		if (!GetStateRegister().RequiresUpdate()) return;

		m_cameras.Update();

		m_point_lights.Update();
		m_spot_lights.Update();
		m_direct_lights.Update();

		m_meshes.Update();
		m_spheres.Update();

		GetStateRegister().Update();
	}
}