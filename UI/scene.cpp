#include "scene.hpp"

#include "vec3.h"

#include <numeric>
#include <random>
#include <array>
#include <numbers>

namespace RZ = RayZath::Engine;

namespace RayZath::UI
{
	Scene::Scene()
		: mr_engine(RZ::Engine::GetInstance())
		, mr_world(mr_engine.GetWorld())
	{}


	void Scene::init()
	{
		createDefaultScene();
	}

	void Scene::render()
	{
		mr_engine.RenderWorld(RZ::Engine::RenderDevice::Default, true, false);
	}

	void Scene::update([[maybe_unused]] const float et)
	{
		auto& cameras = mr_world.Container<RZ::World::ObjectType::Camera>();
		for (uint32_t i = 0; i < cameras.GetCount(); i++)
		{
			auto& camera = cameras[i];

			// auto focus
			const float d1 = camera->GetFocalDistance();
			const auto& p = camera->GetFocalPoint();
			const float d2 = camera->GetDepthBuffer().Value(p.x, p.y);
			if (mr_world.GetStateRegister().IsModified() || std::abs(d1 - d2) > 0.01f * d2)
			{
				camera->Focus(Math::vec2ui32(p.x, p.y));
			}
		}
	}

	// ---- Common Mesh Generators ----
	void Scene::createDefaultScene()
	{
		mr_world.DestroyAll();

		// camera
		auto camera = mr_world.Container<Engine::World::ObjectType::Camera>().Create(
			Engine::ConStruct<Engine::Camera>(
				"camera",
				Math::vec3f32(4.0f, 4.0f, -4.0f), Math::vec3f32{}));
		camera->SetTemporalBlend(0.55f);
		camera->LookAtPoint(Math::vec3f32(.0f, .0f, .0f));
		camera->Focus(camera->GetResolution() / 2);

		// sufrace
		auto surface_mesh = mr_world.GenerateMesh<Engine::World::CommonMesh::Plane>(
			Engine::World::CommonMeshParameters<Engine::World::CommonMesh::Plane>(4, 5.0f, 5.0f));
		surface_mesh->SetName("surface");
		auto surface_material = mr_world.Container<Engine::World::ObjectType::Material>().Create(
			Engine::Material::GenerateMaterial<Engine::Material::Common::Paper>());
		surface_material->SetName("sufrace");
		surface_material->SetColor(Graphics::Color::Palette::Grey);
		auto surface = mr_world.Container<Engine::World::ObjectType::Mesh>().Create(
			Engine::ConStruct<Engine::Mesh>(
				"surface",
				Math::vec3f32(0.0f), Math::vec3f32(0.0f), Math::vec3f32(1.0f),
				surface_mesh,
				surface_material));

		// light
		auto light = mr_world.Container<Engine::World::ObjectType::DirectLight>().Create(
			Engine::ConStruct<Engine::DirectLight>(
				"sun", Math::vec3f32(-1.0f), Graphics::Color::Palette::White, 1500.0f, 0.02f));


		// cube
		auto cube_mesh = mr_world.GenerateMesh<Engine::World::CommonMesh::Cube>(
			Engine::World::CommonMeshParameters<Engine::World::CommonMesh::Cube>{});
		cube_mesh->SetName("cube");
		auto cube_material = mr_world.Container<Engine::World::ObjectType::Material>().Create(
			Engine::Material::GenerateMaterial<Engine::Material::Common::Porcelain>());
		cube_material->SetName("cube");
		cube_material->SetColor(Graphics::Color::Palette::LightGrey);
		auto cube = mr_world.Container<Engine::World::ObjectType::Mesh>().Create(
			Engine::ConStruct<Engine::Mesh>(
				"cube",
				Math::vec3f32(0.0f, 0.5f, 0.0f),
				Math::vec3f32(0.0f),
				Math::vec3f32(1.0f),
				cube_mesh,
				cube_material));
	}
}
