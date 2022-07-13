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
	{
		m_base_scene_path = "D:\\Users\\Greketrotny\\Documents\\RayZath\\Resources\\Scenes\\";
		m_scene_files = {
			"CornelBox\\cornel_box.json",
			"Teapot\\teapot.json",
			"StanfordDragon_100k\\dragon.json",
			"CenterTable\\center_table.json",
			"Bunny\\bunny.json",
			"WoodenCrate\\wooden_crate.json" };
	}



	void Scene::init()
	{
		loadScene(0);
	}
	void Scene::loadScene(size_t scene_id)
	{
		RZAssert(!m_scene_files.empty(), "scene collection is empty");

		if (scene_id < m_scene_files.size())
			mr_world.GetLoader().LoadScene(m_base_scene_path + m_scene_files[scene_id]);

		RZ::Engine::GetInstance().GetRenderConfig().GetTracing().SetMaxDepth(16u);
		RZ::Engine::GetInstance().GetRenderConfig().GetTracing().SetRPP(4u);
	}

	void Scene::render()
	{
		mr_engine.RenderWorld(RZ::Engine::RenderDevice::Default, false, false);
	}

	Math::vec3f polarRotation(const Math::vec3f& v)
	{
		const float theta = acosf(v.Normalized().y);
		const float phi = atan2f(v.z, v.x);
		return { theta, phi, v.Magnitude() };
	}
	Math::vec3f cartesianDirection(const Math::vec3f& polar)
	{
		return Math::vec3f(cosf(polar.y) * sinf(polar.x), cosf(polar.x), sinf(polar.y) * sinf(polar.x)) * polar.z;
	}
	void Scene::update(const float et)
	{
		auto& cameras = mr_world.Container<RZ::World::ObjectType::Camera>();
		for (uint32_t i = 0; i < cameras.GetCount(); i++)
		{
			auto& camera = cameras[i];

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
	template<>
	RZ::Handle<RZ::MeshStructure> Scene::Create<CommonMesh::Plane>(
		const CommonMeshParameters<CommonMesh::Plane>& properties)
	{
		RZAssert(properties.sides >= 3, "shape should have at least 3 sides");

		auto mesh = mr_world.Container<RZ::World::ObjectType::MeshStructure>().Create(
			RZ::ConStruct<RZ::MeshStructure>(
				"generated plane",
				properties.sides, properties.sides, 0, properties.sides - 2));

		// vertices
		const float delta_theta = std::numbers::pi_v<float> *2.0f / properties.sides;
		for (uint32_t i = 0; i < properties.sides; i++)
		{
			const auto angle = delta_theta * i;
			mesh->CreateVertex(
				Math::vec3f32(std::cosf(angle), 0.0f, std::sinf(angle)) *
				Math::vec3f32(properties.width, 0.0f, properties.height));
		}

		// triangles
		for (uint32_t i = 0; i < properties.sides - 2; i++)
		{
			mesh->CreateTriangle({ 0, i + 1, i + 2 });
		}

		return mesh;
	}
	template<>
	RZ::Handle<RZ::MeshStructure> Scene::Create<CommonMesh::Sphere>(
		const CommonMeshParameters<CommonMesh::Sphere>& properties)
	{
		auto mesh = mr_world.Container<RZ::World::ObjectType::MeshStructure>().Create(
			RZ::ConStruct<RZ::MeshStructure>("generated sphere"));

		switch (properties.type)
		{
			case CommonMeshParameters<CommonMesh::Sphere>::Type::UVSphere:
			{
				RZAssert(properties.resolution >= 4, "sphere should have at least 4 subdivisions");

				// vertices
				const float d_theta = std::numbers::pi_v<float> / (properties.resolution / 2);
				const float d_phi = 2.0f * std::numbers::pi_v<float> / properties.resolution;
				for (uint32_t theta = 0; theta < properties.resolution / 2 - 1; theta++)
				{
					for (uint32_t phi = 0; phi < properties.resolution; phi++)
					{
						RZ::Vertex v(0.0f, 1.0f, 0.0f);
						v.RotateX(d_theta * (theta + 1));
						v.RotateY(d_phi * phi);
						mesh->CreateVertex(v);
						if (properties.smooth_shading) mesh->CreateNormal(v);
					}
				}
				const auto top_v_idx = mesh->CreateVertex(Math::vec3f32(0.0f, 1.0f, 0.0f));
				const auto bottom_v_idx = mesh->CreateVertex(Math::vec3f32(0.0f, -1.0f, 0.0f));
				if (properties.smooth_shading)
				{
					mesh->CreateNormal(Math::vec3f32(0.0f, +1.0f, 0.0f));
					mesh->CreateNormal(Math::vec3f32(0.0f, -1.0f, 0.0f));
				}
				

				// triangles
				// top and bottom fan
				for (uint32_t i = 0; i < properties.resolution; i++)
				{
					std::array<uint32_t, 3> top_ids = {
						top_v_idx,
						(i + 1) % properties.resolution,
						i };
					std::array<uint32_t, 3> bottom_ids = {
						bottom_v_idx,
						top_v_idx - properties.resolution + i,
						top_v_idx - properties.resolution + (i + 1) % properties.resolution };
					mesh->CreateTriangle(
						top_ids, 
						RZ::MeshStructure::ids_unused,
						properties.smooth_shading ? top_ids : RZ::MeshStructure::ids_unused);
					mesh->CreateTriangle(
						bottom_ids, 
						RZ::MeshStructure::ids_unused,
						properties.smooth_shading ? bottom_ids : RZ::MeshStructure::ids_unused);
				}
				// middle layers
				for (uint32_t theta = 0; theta < properties.resolution / 2 - 2; theta++)
				{
					for (uint32_t phi = 0; phi < properties.resolution; phi++)
					{
						std::array<uint32_t, 3> ids1 = {
							theta * properties.resolution + phi,
							theta * properties.resolution + (phi + 1) % properties.resolution,
							(theta + 1) * properties.resolution + (phi + 1) % properties.resolution };
						std::array<uint32_t, 3> ids2 = {
							theta * properties.resolution + phi,
							(theta + 1) * properties.resolution + (phi + 1) % properties.resolution,
							(theta + 1) * properties.resolution + phi };
						mesh->CreateTriangle(ids1, RZ::MeshStructure::ids_unused, ids1);
						mesh->CreateTriangle(ids2, RZ::MeshStructure::ids_unused, ids2);
					}
				}
			}
		}

		return mesh;
	}
	template<>
	RZ::Handle<RZ::MeshStructure> Scene::Create<CommonMesh::Cylinder>(
		const CommonMeshParameters<CommonMesh::Cylinder>& properties)
	{
		RZAssert(properties.faces >= 3, "cylinder should have at least 3 faces");

		const auto vertices_num = properties.faces * 2;
		const auto tris_num = (properties.faces - 2) * 2 + properties.faces * 2;
		auto mesh = mr_world.Container<RZ::World::ObjectType::MeshStructure>().Create(
			RZ::ConStruct<RZ::MeshStructure>(
				"generated cylinder",
				vertices_num, 1, vertices_num * 2, tris_num));

		mesh->CreateTexcrd(Math::vec2f32(0.5f, 0.5f));

		// vertices + normals
		const float delta_theta = std::numbers::pi_v<float> *2.0f / properties.faces;
		for (uint32_t i = 0; i < properties.faces; i++)
		{
			const auto angle = delta_theta * i;
			mesh->CreateVertex(Math::vec3f32(std::cosf(angle), -1.0f, std::sinf(angle)));
			mesh->CreateVertex(Math::vec3f32(std::cosf(angle), +1.0f, std::sinf(angle)));

			if (properties.smooth_shading)
				mesh->CreateNormal(Math::vec3f32(1.0f, 0.0f, 0.0f).RotatedY(angle));
		}

		auto vertex_idx = [&vertices_num](const uint32_t idx)
		{
			return idx % vertices_num;
		};

		// triangles
		for (uint32_t i = 0; i < properties.faces - 2; i++)
		{
			// bottom
			mesh->CreateTriangle({
				0,
				vertex_idx((i + 1) * 2),
				vertex_idx((i + 2) * 2) });
			// top
			mesh->CreateTriangle({
				1,
				vertex_idx((i + 2) * 2 + 1),
				vertex_idx((i + 1) * 2 + 1) });
		}
		if (properties.smooth_shading)
		{
			for (uint32_t i = 0; i < properties.faces; i++)
			{
				// side
				mesh->CreateTriangle({
					vertex_idx(i * 2),
					vertex_idx(i * 2 + 1),
					vertex_idx((i + 1) * 2 + 1) },
									 { 0, 0, 0 }, { i, i, (i + 1) % properties.faces });
				mesh->CreateTriangle({
					vertex_idx(i * 2),
					vertex_idx((i + 1) * 2 + 1),
					vertex_idx((i + 1) * 2) },
									 { 0, 0, 0 }, { i, (i + 1) % properties.faces, (i + 1) % properties.faces });
			}
		}
		else
		{
			for (uint32_t i = 0; i < properties.faces; i++)
			{
				// side
				mesh->CreateTriangle({
					vertex_idx(i * 2),
					vertex_idx(i * 2 + 1),
					vertex_idx((i + 1) * 2 + 1) });
				mesh->CreateTriangle({
					vertex_idx(i * 2),
					vertex_idx((i + 1) * 2 + 1),
					vertex_idx((i + 1) * 2) });
			}
		}

		return mesh;
	}

	void Scene::generate()
	{
		if (generated) return;
		generated = true;
		srand(time(NULL));

		CommonMeshParameters<CommonMesh::Sphere> parameters;
		parameters.smooth_shading = true;
		parameters.texture_coordinates = false;
		parameters.resolution = 12;

		auto mesh = Create<CommonMesh::Sphere>(parameters);

		std::vector<RZ::Handle<RZ::Material>> materials;
		for (int i = 0; i < 20; i++)
		{
			materials.push_back(mr_world.Container<RZ::World::ObjectType::Material>().Create(
				RZ::ConStruct<RZ::Material>(
					"triangle material",
					Graphics::Color(rand() & 0xFF, rand() & 0xFF, rand() & 0xFF))));
		}

		for (uint32_t i = 0; i < mesh->GetTriangles().GetCount(); i++)
		{
			mesh->GetTriangles()[i].material_id = rand() & 0x3F;
		}

		auto object = mr_world.Container<RZ::World::ObjectType::Mesh>().Create(
			RZ::ConStruct<RZ::Mesh>("generated mesh",
									{}, {}, Math::vec3f32(1.0f), mesh));

		for (uint32_t i = 0; i < object->GetMaterialCapacity(); i++)
		{
			object->SetMaterial(materials[rand() % materials.size()], i);
		}
	}
}