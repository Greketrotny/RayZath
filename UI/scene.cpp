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

				// vertices + normals
				const float d_theta = std::numbers::pi_v<float> / (properties.resolution / 2);
				const float d_phi = 2.0f * std::numbers::pi_v<float> / properties.resolution;
				for (uint32_t theta = 0; theta < properties.resolution / 2 - 1; theta++)
				{
					for (uint32_t phi = 0; phi < properties.resolution; phi++)
					{
						RZ::Vertex v(0.0f, 1.0f, 0.0f);
						const float a_theta = d_theta * (theta + 1);
						const float a_phi = d_phi * phi;
						v.RotateX(a_theta);
						v.RotateY(a_phi);
						mesh->CreateVertex(v);

						if (properties.normals)
						{
							mesh->CreateNormal(v);
						}
					}
				}
				const auto top_v_idx = mesh->CreateVertex(Math::vec3f32(0.0f, 1.0f, 0.0f));
				const auto bottom_v_idx = mesh->CreateVertex(Math::vec3f32(0.0f, -1.0f, 0.0f));
				if (properties.normals)
				{
					mesh->CreateNormal(Math::vec3f32(0.0f, +1.0f, 0.0f));
					mesh->CreateNormal(Math::vec3f32(0.0f, -1.0f, 0.0f));
				}

				// texture coordinates
				uint32_t top_t_idx = 0, bottom_t_idx = 0;
				if (properties.texture_coordinates)
				{
					for (uint32_t theta = 0; theta < properties.resolution / 2 - 1; theta++)
					{
						for (uint32_t phi = 0; phi < properties.resolution; phi++)
						{
							const float a_theta = d_theta * (theta + 1);
							const float a_phi = d_phi * phi;
							mesh->CreateTexcrd(Math::vec2f32(
								a_phi * 0.5f * std::numbers::inv_pi_v<float>,
								1.0f - a_theta * std::numbers::inv_pi_v<float>));
						}
						mesh->CreateTexcrd(Math::vec2f32(
							1.0f,
							1.0f - (d_theta * (theta + 1)) * std::numbers::inv_pi_v<float>));
					}

					for (uint32_t i = 0; i < properties.resolution; i++)
					{
						auto top_idx = mesh->CreateTexcrd(Math::vec2f32(
							i / float(properties.resolution) + (0.5f / properties.resolution),
							1.0f));
						if (i == 0) top_t_idx = top_idx;
					}
					for (uint32_t i = 0; i < properties.resolution; i++)
					{
						auto bottom_idx = mesh->CreateTexcrd(Math::vec2f32(
							i / float(properties.resolution) + (0.5f / properties.resolution),
							0.0f));
						if (i == 0) bottom_t_idx = bottom_idx;
					}
				}

				// triangles
				using triple_index_t = RZ::MeshStructure::triple_index_t;
				// top and bottom fan
				triple_index_t vn_ids_value{}, t_ids_value{};
				for (uint32_t i = 0; i < properties.resolution; i++)
				{
					const triple_index_t& top_v_ids = vn_ids_value = {
						top_v_idx,
						(i + 1) % properties.resolution,
						i };
					const triple_index_t& top_t_ids = properties.texture_coordinates ? t_ids_value = {
						top_t_idx + i,
						i + 1,
						i } : RZ::MeshStructure::ids_unused;
					const triple_index_t& top_n_ids = properties.normals ? top_v_ids : RZ::MeshStructure::ids_unused;
					mesh->CreateTriangle(top_v_ids, top_t_ids, top_n_ids);

					const triple_index_t& bottom_v_ids = vn_ids_value = {
						bottom_v_idx,
						top_v_idx - properties.resolution + i,
						top_v_idx - properties.resolution + (i + 1) % properties.resolution };
					const triple_index_t& bottom_t_ids = properties.texture_coordinates ? t_ids_value = {
						bottom_t_idx + i,
						top_t_idx - properties.resolution + i - 1,
						top_t_idx - properties.resolution + i } : RZ::MeshStructure::ids_unused;
					const triple_index_t& bottom_n_ids = properties.normals ? bottom_v_ids : RZ::MeshStructure::ids_unused;
					mesh->CreateTriangle(bottom_v_ids, bottom_t_ids, bottom_n_ids);
				}
				// middle layers
				for (uint32_t theta = 0; theta < properties.resolution / 2 - 2; theta++)
				{
					for (uint32_t phi = 0; phi < properties.resolution; phi++)
					{
						const triple_index_t& v_ids1 = vn_ids_value = {
							theta * properties.resolution + phi,
							theta * properties.resolution + (phi + 1) % properties.resolution,
							(theta + 1) * properties.resolution + (phi + 1) % properties.resolution };
						const triple_index_t& t_ids1 = properties.texture_coordinates ? t_ids_value = {
							theta * (properties.resolution + 1) + phi,
							theta * (properties.resolution + 1) + (phi + 1),
							(theta + 1) * (properties.resolution + 1) + (phi + 1) } : RZ::MeshStructure::ids_unused;
						const triple_index_t& n_ids1 = properties.normals ? v_ids1 : RZ::MeshStructure::ids_unused;
						mesh->CreateTriangle(v_ids1, t_ids1, n_ids1);

						const triple_index_t& v_ids2 = vn_ids_value = {
							theta * properties.resolution + phi,
							(theta + 1) * properties.resolution + (phi + 1) % properties.resolution,
							(theta + 1) * properties.resolution + phi };
						const triple_index_t& t_ids2 = properties.texture_coordinates ? t_ids_value = {
							theta * (properties.resolution + 1) + phi,
							(theta + 1) * (properties.resolution + 1) + (phi + 1),
							(theta + 1) * (properties.resolution + 1) + phi } : RZ::MeshStructure::ids_unused;
						const triple_index_t& n_ids2 = properties.normals ? v_ids2 : RZ::MeshStructure::ids_unused;
						mesh->CreateTriangle(v_ids2, t_ids2, n_ids2);
					}
				}
				break;
			}
			default:
				RZThrow("failed to generate sphere with unsupported tesselation method");
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

			if (properties.normals)
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
		if (properties.normals)
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
		parameters.normals = true;
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