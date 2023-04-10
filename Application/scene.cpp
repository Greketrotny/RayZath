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
		: mr_engine(RZ::Engine::instance())
		, mr_world(mr_engine.world())
	{}


	void Scene::init()
	{
		createDefaultScene();
	}

	void Scene::render()
	{
		mr_engine.renderWorld(true, false);
	}

	void Scene::update([[maybe_unused]] const float et)
	{
		auto& cameras = mr_world.container<RZ::ObjectType::Camera>();
		for (uint32_t i = 0; i < cameras.count(); i++)
		{
			auto& camera = cameras[i];

			// auto focus
			const float d1 = camera->focalDistance();
			const auto& p = camera->focalPoint();
			const float d2 = camera->depthBuffer().Value(p.x, p.y);
			if (std::abs(d1 - d2) > 0.01f * d2)
			{
				camera->focus(Math::vec2ui32(p.x, p.y));
			}
		}
	}

	// ---- Common Mesh Generators ----
	void Scene::createDefaultScene()
	{
		mr_world.destroyAll();

		// camera
		auto camera = mr_world.container<Engine::ObjectType::Camera>().create(
			Engine::ConStruct<Engine::Camera>(
				"camera",
				Math::vec3f32(4.0f, 4.0f, -4.0f), Math::vec3f32{}));
		camera->temporalBlend(0.55f);
		camera->lookAtPoint(Math::vec3f32(.0f, .0f, .0f));
		camera->focus(camera->resolution() / 2);

		// sufrace
		auto surface_mesh = mr_world.generateMesh<Engine::World::CommonMesh::Plane>(
			Engine::World::CommonMeshParameters<Engine::World::CommonMesh::Plane>(4, 5.0f, 5.0f));
		surface_mesh->name("surface");
		auto surface_material = mr_world.container<Engine::ObjectType::Material>().create(
			Engine::Material::generateMaterial<Engine::Material::Common::Paper>());
		surface_material->name("surface");
		surface_material->color(Graphics::Color::Palette::Grey);
		auto surface = mr_world.container<Engine::ObjectType::Instance>().create(
			Engine::ConStruct<Engine::Instance>(
				"surface",
				Math::vec3f32(0.0f), Math::vec3f32(0.0f), Math::vec3f32(1.0f),
				surface_mesh,
				surface_material));

		// light
		mr_world.container<Engine::ObjectType::DirectLight>()->add(Engine::ConStruct<Engine::DirectLight>(
			"sun", Math::vec3f32(-1.0f), Graphics::Color::Palette::White, 1500.0f, 0.02f));


		// cube
		auto cube_mesh = mr_world.generateMesh<Engine::World::CommonMesh::Cube>(
			Engine::World::CommonMeshParameters<Engine::World::CommonMesh::Cube>{});
		cube_mesh->name("cube");
		auto cube_material = mr_world.container<Engine::ObjectType::Material>().create(
			Engine::Material::generateMaterial<Engine::Material::Common::Porcelain>());
		cube_material->name("cube");
		cube_material->color(Graphics::Color::Palette::LightGrey);
		auto cube = mr_world.container<Engine::ObjectType::Instance>().create(
			Engine::ConStruct<Engine::Instance>(
				"cube",
				Math::vec3f32(0.0f, 0.5f, 0.0f),
				Math::vec3f32(0.0f),
				Math::vec3f32(1.0f),
				cube_mesh,
				cube_material));
	}
}
