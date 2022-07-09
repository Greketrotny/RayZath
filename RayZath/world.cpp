#include "world.h"
#include "json_loader.h"
#include "loader.h"

#include <string_view>
#include <fstream>
#include <sstream>

namespace RayZath::Engine
{
	// ~~~~~~~~ [CLASS] World ~~~~~~~~
	World::World()
		: Updatable(nullptr)
		, m_containers(
			ObjectContainer<Texture>(this),
			ObjectContainer<NormalMap>(this),
			ObjectContainer<MetalnessMap>(this),
			ObjectContainer<RoughnessMap>(this),
			ObjectContainer<EmissionMap>(this),
			ObjectContainer<Material>(this),
			ObjectContainer<MeshStructure>(this),
			ObjectContainer<Camera>(this),
			ObjectContainer<SpotLight>(this),
			ObjectContainer<DirectLight>(this),
			ObjectContainerWithBVH<Mesh>(this),
			ObjectContainer<Group>(this))
		, m_material(
			this,
			ConStruct<Material>(
				"world_material",
				Graphics::Color(0xFF, 0xFF, 0xFF, 0x00),
				0.0f, 0.0f, 0.0f, 1.0f, 0.0f))
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
		Container<ObjectType::Texture>().DestroyAll();
		Container<ObjectType::NormalMap>().DestroyAll();
		Container<ObjectType::MetalnessMap>().DestroyAll();
		Container<ObjectType::RoughnessMap>().DestroyAll();
		Container<ObjectType::EmissionMap>().DestroyAll();

		Container<ObjectType::Material>().DestroyAll();
		Container<ObjectType::MeshStructure>().DestroyAll();

		Container<ObjectType::Camera>().DestroyAll();

		Container<ObjectType::SpotLight>().DestroyAll();
		Container<ObjectType::DirectLight>().DestroyAll();

		Container<ObjectType::Mesh>().DestroyAll();
		Container<ObjectType::Group>().DestroyAll();
	}

	void World::Update()
	{
		if (!GetStateRegister().RequiresUpdate()) return;


		Container<ObjectType::Texture>().Update();
		Container<ObjectType::NormalMap>().Update();
		Container<ObjectType::MetalnessMap>().Update();
		Container<ObjectType::RoughnessMap>().Update();
		Container<ObjectType::EmissionMap>().Update();

		Container<ObjectType::Material>().Update();
		Container<ObjectType::MeshStructure>().Update();

		Container<ObjectType::Camera>().Update();

		Container<ObjectType::SpotLight>().Update();
		Container<ObjectType::DirectLight>().Update();

		Container<ObjectType::Mesh>().Update();
		Container<ObjectType::Group>().Update();

		GetStateRegister().Update();
	}
}