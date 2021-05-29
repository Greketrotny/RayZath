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
			ObjectContainer<EmittanceMap>(this, 16u),
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
				Graphics::Color(0xFF, 0xFF, 0xFF, 0xFF),
				0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 0.0f)),
		m_default_material(
			this, 
			ConStruct<Material>(
				Graphics::Color(0xFF, 0xFF, 0xFF, 0x00),
				0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f))
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
		Container<ContainerType::EmittanceMap>().DestroyAll();

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

	Handle<Material> World::GenerateGlassMaterial(const Handle<Texture>& texture)
	{
		return Container<ContainerType::Material>().Create(ConStruct<Material>(
			Graphics::Color::Palette::White,
			0.0f, 0.0f, 1.0f, 1.5f, 0.0f, 0.0f, texture));
	}
	Handle<Material> World::GenerateMirrorMaterial(const Handle<Texture>& texture)
	{
		return Container<ContainerType::Material>().Create(ConStruct<Material>(
			Graphics::Color::Palette::White,
			1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, texture));
	}
	Handle<Material> World::GenerateDiffuseMaterial(const Handle<Texture>& texture)
	{
		return Container<ContainerType::Material>().Create(ConStruct<Material>(
			Graphics::Color::Palette::White,
			0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, texture));
	}
	Handle<Material> World::GenerateGlossyMaterial(const Handle<Texture>& texture)
	{
		return Container<ContainerType::Material>().Create(ConStruct<Material>(
			Graphics::Color::Palette::White,
			1.0f, 0.01f, 0.0f, 1.0f, 0.0f, 0.0f, texture));
	}

	void World::Update()
	{
		if (!GetStateRegister().RequiresUpdate()) return;

		
		Container<ContainerType::Texture>().Update();
		Container<ContainerType::NormalMap>().Update();
		Container< ContainerType::EmittanceMap>().Update();

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