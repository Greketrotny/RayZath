#include "world.h"

namespace RayZath
{
	// ~~~~~~~~ [STRUCT] Containers ~~~~~~~~
	World::Containers::Containers(
		Updatable* parent,
		uint32_t cameras_capacity,
		uint32_t lights_capacity,
		uint32_t renderables_capacity)
		: ObjectContainer<Texture>(parent, 16u)
		, ObjectContainer<Material>(parent, 16u)
		, ObjectContainer<MeshStructure>(parent, 1024u)
		, ObjectContainer<Camera>(parent, cameras_capacity)
		, ObjectContainer<PointLight>(parent, lights_capacity)
		, ObjectContainer<SpotLight>(parent, lights_capacity)
		, ObjectContainer<DirectLight>(parent, lights_capacity)
		, ObjectContainerWithBVH<Mesh>(parent, renderables_capacity)
		, ObjectContainerWithBVH<Sphere>(parent, renderables_capacity)
		, ObjectContainer<Plane>(parent, renderables_capacity)
	{}
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



	// ~~~~~~~~ [CLASS] World ~~~~~~~~
	World::World(
		const uint32_t& maxCamerasCount,
		const uint32_t& maxLightsCount,
		const uint32_t& maxRenderObjectsCount)
		: Updatable(nullptr)
		, m_containers(this, maxCamerasCount, maxLightsCount, maxRenderObjectsCount)
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
		Container<Texture>().DestroyAll();
		Container<Material>().DestroyAll();
		Container<MeshStructure>().DestroyAll();

		Container<Camera>().DestroyAll();

		Container<PointLight>().DestroyAll();
		Container<SpotLight>().DestroyAll();
		Container<DirectLight>().DestroyAll();

		Container<Mesh>().DestroyAll();
		Container<Sphere>().DestroyAll();
		Container<Plane>().DestroyAll();
	}

	Handle<Material> World::GenerateGlassMaterial(const Handle<Texture>& texture)
	{
		return Container<Material>().Create(ConStruct<Material>(
			Graphics::Color::Palette::White,
			0.0f, 0.0f, 1.0f, 1.5f, 0.0f, 0.0f, texture));
	}
	Handle<Material> World::GenerateMirrorMaterial(const Handle<Texture>& texture)
	{
		return Container<Material>().Create(ConStruct<Material>(
			Graphics::Color::Palette::White,
			1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, texture));
	}
	Handle<Material> World::GenerateDiffuseMaterial(const Handle<Texture>& texture)
	{
		return Container<Material>().Create(ConStruct<Material>(
			Graphics::Color::Palette::White,
			0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, texture));
	}
	Handle<Material> World::GenerateGlossyMaterial(const Handle<Texture>& texture)
	{
		return Container<Material>().Create(ConStruct<Material>(
			Graphics::Color::Palette::White,
			1.0f, 0.01f, 0.0f, 1.0f, 0.0f, 0.0f, texture));
	}

	void World::Update()
	{
		if (!GetStateRegister().RequiresUpdate()) return;

		Container<Texture>().Update();
		Container<Material>().Update();
		Container<MeshStructure>().Update();

		Container<Camera>().Update();

		Container<PointLight>().Update();
		Container<SpotLight>().Update();
		Container<DirectLight>().Update();

		Container<Mesh>().Update();
		Container<Sphere>().Update();
		Container<Plane>().Update();

		GetStateRegister().Update();
	}
}