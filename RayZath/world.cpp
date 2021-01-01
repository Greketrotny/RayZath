#include "world.h"

namespace RayZath
{
	// ~~~~~~~~ [CLASS] World ~~~~~~~~
	World::World(
		const uint32_t& maxCamerasCount,
		const uint32_t& maxLightsCount,
		const uint32_t& maxRenderObjectsCount)
		: Updatable(nullptr)
		, m_textures(this, 16u)
		, m_materials(this, 16u)
		, m_mesh_structures(this, 1024u)
		, m_cameras(this, maxCamerasCount)
		, m_point_lights(this, maxLightsCount)
		, m_spot_lights(this, maxLightsCount)
		, m_direct_lights(this, maxLightsCount)
		, m_meshes(this, maxRenderObjectsCount)
		, m_spheres(this, maxRenderObjectsCount)
		, m_planes(this, maxRenderObjectsCount)
		, m_material(
			this,
			ConStruct<Material>(
				Graphics::Color(0xFF, 0xFF, 0xFF, 0xFF),
				0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 0.0f)),
		m_default_material(
			this, 
			ConStruct<Material>(
				Graphics::Color(0xFF, 0xFF, 0xFF, 0x00),
				0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f))
	{}
	World::~World()
	{}

	ObjectContainer<Texture>& World::GetTextures()
	{
		return m_textures;
	}
	const ObjectContainer<Texture>& World::GetTextures() const
	{
		return m_textures;
	}
	ObjectContainer<Material>& World::GetMaterials()
	{
		return m_materials;
	}
	const ObjectContainer<Material>& World::GetMaterials() const
	{
		return m_materials;
	}
	ObjectContainer<MeshStructure>& World::GetMeshStructures()
	{
		return m_mesh_structures;
	}
	const ObjectContainer<MeshStructure>& World::GetMeshStructures() const
	{
		return m_mesh_structures;
	}

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
	ObjectContainer<Plane>& World::GetPlanes()
	{
		return m_planes;
	}
	const ObjectContainer<Plane>& World::GetPlanes() const
	{
		return m_planes;
	}

	Material& World::GetMaterial()
	{
		return m_material;
	}
	const Material& World::GetMaterial() const
	{
		return m_material;
	}
	Material& World::GetDefaultMaterial()
	{
		return m_default_material;
	}
	const Material& World::GetDefaultMaterial() const
	{
		return m_default_material;
	}

	void World::DestroyAllComponents()
	{
		m_materials.DestroyAll();
		m_mesh_structures.DestroyAll();

		m_cameras.DestroyAll();

		m_point_lights.DestroyAll();
		m_spot_lights.DestroyAll();
		m_direct_lights.DestroyAll();

		m_meshes.DestroyAll();
		m_spheres.DestroyAll();
	}

	void World::Update()
	{
		if (!GetStateRegister().RequiresUpdate()) return;

		m_materials.Update();
		m_mesh_structures.Update();

		m_cameras.Update();

		m_point_lights.Update();
		m_spot_lights.Update();
		m_direct_lights.Update();

		m_meshes.Update();
		m_spheres.Update();
		m_planes.Update();

		GetStateRegister().Update();
	}
}