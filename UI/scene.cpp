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

	//RZ::Handle<RZ::Mesh> Scene::CreateCube(
	//	RZ::World& world,
	//	RZ::ConStruct<RZ::Mesh> conStruct)
	//{
	//	// create mesh structure
	//	RZ::Handle<RZ::MeshStructure> structure = world.Container<RZ::World::ObjectType::MeshStructure>().Create(
	//		RZ::ConStruct<RZ::MeshStructure>("cube", 8u, 4u, 0u, 12u));


	//	/*
	//			vertices				texcrds

	//				   1 --------- 2
	//				  /:          /|		1 --------- 2
	//				 / :         / |		|			|
	//				0 --------- 3  |		|			|
	//				|  :		|  |        |			|
	//				|  5 . . . .|. 6        |			|
	//				| /		    | /         0 --------- 3
	//		^  >	|/		    |/
	//		| /		4 --------- 7
	//		|/
	//		o----->
	//	*/

	//	// vertices
	//	structure->CreateVertex(-1.0f, 1.0f, -1.0f);
	//	structure->CreateVertex(-1.0f, 1.0f, 1.0f);
	//	structure->CreateVertex(1.0f, 1.0f, 1.0f);
	//	structure->CreateVertex(1.0f, 1.0f, -1.0f);
	//	structure->CreateVertex(-1.0f, -1.0f, -1.0f);
	//	structure->CreateVertex(-1.0f, -1.0f, 1.0f);
	//	structure->CreateVertex(1.0f, -1.0f, 1.0f);
	//	structure->CreateVertex(1.0f, -1.0f, -1.0f);

	//	// texcrds
	//	structure->CreateTexcrd(0.0f, 0.0f);
	//	structure->CreateTexcrd(0.0f, 1.0f);
	//	structure->CreateTexcrd(1.0f, 1.0f);
	//	structure->CreateTexcrd(1.0f, 0.0f);

	//	// triangles
	//	// top
	//	structure->CreateTriangle({ 1, 2, 0 }, { 1, 2, 0 });
	//	structure->CreateTriangle({ 3, 0, 2 }, { 3, 0, 2 });
	//	// bottom
	//	structure->CreateTriangle({ 4, 7, 5 }, { 1, 2, 0 });
	//	structure->CreateTriangle({ 6, 5, 7 }, { 3, 0, 2 });
	//	// front
	//	structure->CreateTriangle({ 0, 3, 4 }, { 1, 2, 0 });
	//	structure->CreateTriangle({ 7, 4, 3 }, { 3, 0, 2 });
	//	// back
	//	structure->CreateTriangle({ 2, 1, 6 }, { 1, 2, 0 });
	//	structure->CreateTriangle({ 5, 6, 1 }, { 3, 0, 2 });
	//	// right
	//	structure->CreateTriangle({ 3, 2, 7 }, { 1, 2, 0 });
	//	structure->CreateTriangle({ 6, 7, 2 }, { 3, 0, 2 });
	//	// left
	//	structure->CreateTriangle({ 1, 0, 5 }, { 1, 2, 0 });
	//	structure->CreateTriangle({ 4, 5, 0 }, { 3, 0, 2 });

	//	conStruct.mesh_structure = structure;
	//	return world.Container<RZ::World::ObjectType::Mesh>().Create(conStruct);
	//}
	/*void Scene::CreateTessellatedSphere(
		RZ::World* world,
		const RZ::ConStruct<RZ::Mesh>& conStruct,
		const uint32_t& res)
	{
		// [>] create Mesh object
		RZ::Mesh* mesh = world->GetMeshes().Create(conStruct);
		mesh->GetMeshStructure()->Reset(10000u, 2u, 10000u, 10000u);

		// [>] Create vertices
		// middle layer vertices
		for (uint32_t i = 1; i < res - 1; ++i)
		{
			for (uint32_t j = 0; j < res; ++j)
			{
				RZ::Vertex v(0.0f, 1.0f, 0.0f);
				v.RotateX(((float)i / (float)(res - 1)) * Math::constants<float>::pi);
				v.RotateY(((float)j / (float)(res)) * Math::constants<float>::pi * 2.0f);
				mesh->GetMeshStructure()->CreateVertex(v);
				mesh->GetMeshStructure()->CreateNormal(v);
			}
		}

		// top and bottom vertices
		Math::vec3f* vTop = mesh->GetMeshStructure()->CreateVertex(0.0f, 1.0f, 0.0f);
		Math::vec3f* vBottom = mesh->GetMeshStructure()->CreateVertex(0.0f, -1.0f, 0.0f);

		Math::vec3f* vNTop = mesh->GetMeshStructure()->CreateNormal(0.0f, 1.0f, 0.0f);
		Math::vec3f* vNBottom = mesh->GetMeshStructure()->CreateNormal(0.0f, -1.0f, 0.0f);


		// [>] Create triangles
		// hat and foot
		for (uint32_t i = 0; i < res; ++i)
		{
			mesh->GetMeshStructure()->CreateTriangle(
				vTop,
				&mesh->GetMeshStructure()->GetVertices()[(i + 1) % res],
				&mesh->GetMeshStructure()->GetVertices()[i],
				nullptr, nullptr, nullptr,
				vNTop,
				&mesh->GetMeshStructure()->GetNormals()[(i + 1) % res],
				&mesh->GetMeshStructure()->GetNormals()[i]);
			mesh->GetMeshStructure()->CreateTriangle(
				vBottom,
				&mesh->GetMeshStructure()->GetVertices()[res * (res - 3) + i % res],
				&mesh->GetMeshStructure()->GetVertices()[res * (res - 3) + (i + 1) % res],
				nullptr, nullptr, nullptr,
				vNBottom,
				&mesh->GetMeshStructure()->GetNormals()[res * (res - 3) + i % res],
				&mesh->GetMeshStructure()->GetNormals()[res * (res - 3) + (i + 1) % res]);
		}

		// middle layers
		for (uint32_t i = 0; i < res - 3; ++i)
		{
			for (uint32_t j = 0; j < res; ++j)
			{
				mesh->GetMeshStructure()->CreateTriangle(
					&mesh->GetMeshStructure()->GetVertices()[i * res + j],
					&mesh->GetMeshStructure()->GetVertices()[(i + 1) * res + (j + 1) % res],
					&mesh->GetMeshStructure()->GetVertices()[(i + 1) * res + j],
					nullptr, nullptr, nullptr,
					&mesh->GetMeshStructure()->GetNormals()[i * res + j],
					&mesh->GetMeshStructure()->GetNormals()[(i + 1) * res + (j + 1) % res],
					&mesh->GetMeshStructure()->GetNormals()[(i + 1) * res + j]);

				mesh->GetMeshStructure()->CreateTriangle(
					&mesh->GetMeshStructure()->GetVertices()[i * res + j],
					&mesh->GetMeshStructure()->GetVertices()[i * res + (j + 1) % res],
					&mesh->GetMeshStructure()->GetVertices()[(i + 1) * res + (j + 1) % res],
					nullptr, nullptr, nullptr,
					&mesh->GetMeshStructure()->GetNormals()[i * res + j],
					&mesh->GetMeshStructure()->GetNormals()[i * res + (j + 1) % res],
					&mesh->GetMeshStructure()->GetNormals()[(i + 1) * res + (j + 1) % res]);
			}
		}


		// triangle coloring
		for (uint32_t i = 0; i < mesh->GetMeshStructure()->GetTriangles().GetCapacity(); ++i)
		{
			mesh->GetMeshStructure()->GetTriangles()[i].color =
				Graphics::Color(
					rand() % 63 + 192,
					rand() % 63 + 192,
					rand() % 63 + 192,
					0x00);
		}
	}*/
}