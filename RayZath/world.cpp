#include "world.h"
#include "json_loader.h"
#include "loader.h"

#include <string_view>
#include <fstream>
#include <sstream>

namespace RayZath
{
	// ~~~~~~~~ [CLASS] World ~~~~~~~~
	World::World()
		: Updatable(nullptr)
		, m_containers(
			ObjectContainer<Texture>(this),
			ObjectContainer<NormalMap>(this),
			ObjectContainer<MetalnessMap>(this),
			ObjectContainer<SpecularityMap>(this),
			ObjectContainer<RoughnessMap>(this),
			ObjectContainer<EmissionMap>(this),
			ObjectContainer<Material>(this),
			ObjectContainer<MeshStructure>(this),
			ObjectContainer<Camera>(this),
			ObjectContainer<PointLight>(this),
			ObjectContainer<SpotLight>(this),
			ObjectContainer<DirectLight>(this),
			ObjectContainerWithBVH<Mesh>(this),
			ObjectContainerWithBVH<Sphere>(this),
			ObjectContainer<Plane>(this))
		, m_material(
			this,
			ConStruct<Material>(
				"world_material",
				Graphics::Color(0xFF, 0xFF, 0xFF, 0x00),
				0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f))
		, m_default_material(
			this,
			ConStruct<Material>(
				"world_default_material",
				Graphics::Color::Palette::LightGrey))
		, mp_loader(new Loader(*this))
	{}

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

	Loader& World::GetLoader()
	{
		return *mp_loader;
	}
	const Loader& World::GetLoader() const
	{
		return *mp_loader;
	}

	void World::DestroyAll()
	{
		Container<ContainerType::Texture>().DestroyAll();
		Container<ContainerType::NormalMap>().DestroyAll();
		Container<ContainerType::MetalnessMap>().DestroyAll();
		Container<ContainerType::SpecularityMap>().DestroyAll();
		Container<ContainerType::RoughnessMap>().DestroyAll();
		Container<ContainerType::EmissionMap>().DestroyAll();

		Container<ContainerType::Material>().DestroyAll();
		Container<ContainerType::MeshStructure>().DestroyAll();

		Container<ContainerType::Camera>().DestroyAll();

		Container<ContainerType::PointLight>().DestroyAll();
		Container<ContainerType::SpotLight>().DestroyAll();
		Container<ContainerType::DirectLight>().DestroyAll();

		Container<ContainerType::Mesh>().DestroyAll();
		Container<ContainerType::Sphere>().DestroyAll();
		Container<ContainerType::Plane>().DestroyAll();
	}

	void World::Update()
	{
		if (!GetStateRegister().RequiresUpdate()) return;


		Container<ContainerType::Texture>().Update();
		Container<ContainerType::NormalMap>().Update();
		Container<ContainerType::MetalnessMap>().Update();
		Container<ContainerType::SpecularityMap>().Update();
		Container<ContainerType::RoughnessMap>().Update();
		Container<ContainerType::EmissionMap>().Update();

		Container<ContainerType::Material>().Update();
		Container<ContainerType::MeshStructure>().Update();

		Container<ContainerType::Camera>().Update();

		Container<ContainerType::PointLight>().Update();
		Container<ContainerType::SpotLight>().Update();
		Container<ContainerType::DirectLight>().Update();

		Container<ContainerType::Mesh>().Update();
		Container<ContainerType::Sphere>().Update();
		Container<ContainerType::Plane>().Update();

		GetStateRegister().Update();
	}
}