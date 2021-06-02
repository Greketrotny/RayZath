#include "world.h"

namespace RayZath
{
	// ~~~~~~~~ [CLASS] World ~~~~~~~~
	World::World(
		const uint32_t& maxCamerasCount,
		const uint32_t& maxLightsCount,
		const uint32_t& maxRenderObjectsCount)
		: Updatable(nullptr)
		, m_containers(
			ObjectContainer<Texture>(this, 16u),
			ObjectContainer<NormalMap>(this, 16u),
			ObjectContainer<MetalicMap>(this, 16u),
			ObjectContainer<SpecularMap>(this, 16u),
			ObjectContainer<RoughnessMap>(this, 16u),
			ObjectContainer<EmissionMap>(this, 16u),
			ObjectContainer<Material>(this, 16u),
			ObjectContainer<MeshStructure>(this, 1024u),
			ObjectContainer<Camera>(this, maxCamerasCount),
			ObjectContainer<PointLight>(this, maxLightsCount),
			ObjectContainer<SpotLight>(this, maxLightsCount),
			ObjectContainer<DirectLight>(this, maxLightsCount),
			ObjectContainerWithBVH<Mesh>(this, maxRenderObjectsCount),
			ObjectContainerWithBVH<Sphere>(this, maxRenderObjectsCount),
			ObjectContainer<Plane>(this, maxRenderObjectsCount))
		, m_material(
			this,
			ConStruct<Material>(
				Graphics::Color(0xFF),
				0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f)),
		m_default_material(
			this, 
			ConStruct<Material>(
				Graphics::Color::Palette::LightGrey))
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

	void World::DestroyAll()
	{
		Container<ContainerType::Texture>().DestroyAll();
		Container<ContainerType::NormalMap>().DestroyAll();
		Container<ContainerType::MetalicMap>().DestroyAll();
		Container<ContainerType::SpecularMap>().DestroyAll();
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

	Handle<Material> World::GenerateGlassMaterial()
	{
		return Container<ContainerType::Material>().Create(ConStruct<Material>(
			Graphics::Color::Palette::White,
			0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 1.5f, 0.0f));
	}
	Handle<Material> World::GenerateMirrorMaterial()
	{
		return Container<ContainerType::Material>().Create(ConStruct<Material>(
			Graphics::Color::Palette::White,
			0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f));
	}
	Handle<Material> World::GenerateDiffuseMaterial()
	{
		return Container<ContainerType::Material>().Create(ConStruct<Material>(
			Graphics::Color::Palette::White,
			0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f));
	}

	void World::Update()
	{
		if (!GetStateRegister().RequiresUpdate()) return;

		
		Container<ContainerType::Texture>().Update();
		Container<ContainerType::NormalMap>().Update();
		Container<ContainerType::MetalicMap>().Update();
		Container<ContainerType::SpecularMap>().Update();
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