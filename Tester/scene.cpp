#include "scene.h"
#include "application.h"
#include "loader.h"

#include <numeric>
#include <random>
#include <array>

Graphics::Bitmap GenerateColorBitmap()
{
	unsigned int resolution = 32;
	Graphics::Bitmap bitmap(resolution * 8, resolution);

	unsigned char hs = 0xCC;
	unsigned char ls = 0x44;
	//std::vector<Graphics::Color> colors{
	//	Graphics::Color(hs, ls, ls),	// red
	//	Graphics::Color(ls, hs, ls),	// green
	//	Graphics::Color(ls, ls, hs),	// blue
	//	Graphics::Color(ls, hs, hs),	// cyan
	//	Graphics::Color(hs, hs, ls),	// yellow
	//	Graphics::Color(hs, ls, hs),	// magenta
	//	Graphics::Color(hs, hs, hs),	// white
	//	Graphics::Color(ls, ls, ls),	// dark grey
	//};
	std::vector<Graphics::Color> colors{
		Graphics::Color(hs, hs, hs),	// white
		Graphics::Color(hs, hs, hs),	// white
		Graphics::Color(hs, ls, ls),	// red
		Graphics::Color(ls, hs, ls),	// green
		Graphics::Color(hs, hs, hs),	// white
		Graphics::Color(hs, ls, hs),	// magenta
		Graphics::Color(hs, hs, hs),	// white
		Graphics::Color(ls, ls, ls),	// dark grey
	};

	for (unsigned int i = 0; i < 8; ++i)
	{
		for (unsigned int x = 0; x < resolution; ++x)
		{
			for (unsigned int y = 0; y < resolution; ++y)
			{
				bitmap.Value(resolution * i + x, y) = colors[i];
				//if ((x % 2 == 0) ^ (y % 2 == 0)) bitmap.SetPixel(resolution * i + x, y, colors[i]);
				//else bitmap.SetPixel(resolution * i + x, y, Graphics::Color::Mix(colors[i], Graphics::Color(0x00, 0x00, 0x00)));

				//if (x == 2 && y == 2)
				//	bitmap.SetPixel(x, y, Graphics::Color(0x00, 0x00, 0xFF));
			}
		}
	}

	return bitmap;
}

Graphics::Bitmap GenerateBitmap(
	const uint32_t resolution,
	const Graphics::Color& color1,
	const Graphics::Color& color2)
{
	Graphics::Bitmap bitmap(resolution, resolution);

	for (uint32_t x = 0u; x < resolution; ++x)
	{
		for (uint32_t y = 0u; y < resolution; ++y)
		{
			if ((x % 2 == 0) ^ (y % 2 == 0))
				bitmap.Value(x, y) = color1;
			else
				bitmap.Value(x, y) = color2;
		}
	}

	//bitmap.Value(0u, 0u) = Graphics::Color::Palette::Red;
	//bitmap.Value(bitmap.GetHeight() - 1u, 0u) = Graphics::Color::Palette::Green;
	//bitmap.Value(0u, bitmap.GetWidth() - 1u) = Graphics::Color::Palette::Blue;
	//bitmap.Value(bitmap.GetHeight() - 1u, bitmap.GetWidth() - 1u) = Graphics::Color::Palette::Black;

	return bitmap;
}

Graphics::Buffer2D<float> GenerateEmissionMap(
	const uint32_t resolution,
	const float& force)
{
	Graphics::Buffer2D<float> emittance_map(resolution, resolution);

	for (uint32_t x = 0u; x < resolution; x++)
	{
		for (uint32_t y = 0u; y < resolution; y++)
		{
			if ((x % 2 == 0) ^ (y % 2 == 0))
				emittance_map.Value(x, y) = 0.0f;
			else
				emittance_map.Value(x, y) = force;
		}
	}

	return emittance_map;
}


namespace Tester
{
	Scene::Scene(Application& app)
		: mr_app(app)
		, mr_engine(RZ::Engine::GetInstance())
		, mr_world(mr_engine.GetWorld())
	{
		m_base_scene_path = "D:\\Users\\Greketrotny\\Documents\\RayZath\\Resources\\Scenes\\";
		m_scene_files = {
			"CornelBox\\cornel_box.json",
			"Teapot\\teapot.json",
			"StanfordDragon_100k\\stanford-dragon.json",
			"CenterTable\\center_table.json",
			"Bunny\\bunny.json" };
	}

	void Scene::Init()
	{
		LoadScene(0);
	}

	void Scene::LoadScene(size_t scene_id)
	{
		if (m_scene_files.empty())
			ThrowException("scene collection is empty");

		if (scene_id < m_scene_files.size())
		{
			mr_world.GetLoader().LoadScene(m_base_scene_path + m_scene_files[scene_id]);
			m_camera = mr_world.Container<RZ::World::ContainerType::Camera>()[0];
		}

		RZ::Engine::GetInstance().GetRenderConfig().GetLightSampling().SetSpotLight(4);

		std::default_random_engine re(1234u);

		const int count = 0;
		const float spread = 1.5f;
		for (int x = 0; x < count; x++)
		{
			for (int z = 0; z < count; z++)
			{
				const float x_pos = x - (count - 1) / 2.0f;
				const float z_pos = z - (count - 1) / 2.0f;
				float y_pos = std::uniform_real_distribution<float>(1.0f, 10.0f)(re);

				if (x_pos < 1.5f && x_pos > -1.5f &&
					y_pos < 2.5f && y_pos > -0.5f &&
					z_pos < 3.0f && z_pos > -3.0f)
					float y_pos = std::uniform_real_distribution<float>(2.5f, 10.0f)(re);


				Graphics::Color color(
					std::uniform_int_distribution(0, 256)(re),
					std::uniform_int_distribution(0, 256)(re),
					std::uniform_int_distribution(0, 256)(re));

				mr_world.Container<RZ::World::ContainerType::PointLight>().Create(
					RZ::ConStruct<RZ::PointLight>("light" + std::to_string(x * count + z),
						Math::vec3f(
							x_pos * spread,
							y_pos,
							z_pos * spread),
						color, 0.25f, 10.0f));
				/*mr_world.Container<RZ::World::ContainerType::SpotLight>().Create(
					RZ::ConStruct<RZ::SpotLight>("light" + std::to_string(x * count + z),
						Math::vec3f(
							x_pos * spread,
							y_pos,
							z_pos * spread),
						Math::vec3f(0.0f, -1.0f, 0.0f),
						color, 0.2f, 10.0f, 0.5f, 0.1f));*/
			}
		}
	}

	void Scene::Render()
	{
		mr_engine.RenderWorld(RZ::Engine::RenderDevice::Default, true, false);
	}
	const Graphics::Bitmap& Scene::GetRender()
	{
		return m_camera->GetImageBuffer();
	}
	void Scene::ResizeRender(uint32_t width, uint32_t height)
	{
		m_camera->Resize(Math::vec2ui32(width, height));
	}
	void Scene::Update(const float et)
	{
		const float d1 = m_camera->GetFocalDistance();

		const WAF::Point p = mr_app.m_ui.GetRenderWindow()->focal_point;
		const float d2 = m_camera->GetDepthBuffer().Value(p.x, p.y);
		if (mr_world.GetStateRegister().IsModified() || std::abs(d1 - d2) > 0.01f * d2)
		{
			m_camera->Focus(Math::vec2ui32(p.x, p.y));
		}

		if (mr_world.Container<RZ::World::ContainerType::SpotLight>().GetCount() > 0u)
		{
			if (mr_world.GetStateRegister().IsModified())
			{
				auto light = mr_world.Container<RZ::World::ContainerType::SpotLight>()[0];
				light->SetDirection(-light->GetPosition());
			}
		}

		/*if (mr_world.Container<RZ::World::ContainerType::PointLight>().GetCount() > 0u)
		{
			auto& sun = mr_world.Container<RZ::World::ContainerType::PointLight>()[0];
			auto dir = sun->GetDirection();
			dir.RotateY(0.0001f * et);
			sun->SetDirection(dir);
		}*/

		/*auto pos = m_camera->GetPosition();
		pos += m_camera->GetCoordSystem().GetXAxis() * 0.02f;
		m_camera->SetPosition(pos);
		m_camera->LookAtPoint(Math::vec3f(0.0f));*/

		return;

		float speed = 0.001f * et;

		Math::vec3f rot = cube->GetTransformation().GetRotation();
		rot += Math::vec3f(1.0f * speed, 0.43f * speed, 0.0f);
		cube->SetRotation(rot);
	}

	RZ::Handle<RZ::Mesh> Scene::CreateCube(
		RZ::World& world,
		RZ::ConStruct<RZ::Mesh> conStruct)
	{
		// create mesh structure
		RZ::Handle<RZ::MeshStructure> structure = world.Container<RZ::World::ContainerType::MeshStructure>().Create(
			RZ::ConStruct<RZ::MeshStructure>("cube", 8u, 4u, 0u, 12u));


		/*
				vertices				texcrds

					   1 --------- 2
					  /:          /|		1 --------- 2
					 / :         / |		|			|
					0 --------- 3  |		|			|
					|  :		|  |        |			|
					|  5 . . . .|. 6        |			|
					| /		    | /         0 --------- 3
			^  >	|/		    |/
			| /		4 --------- 7
			|/
			o----->
		*/

		// vertices
		structure->CreateVertex(-1.0f, 1.0f, -1.0f);
		structure->CreateVertex(-1.0f, 1.0f, 1.0f);
		structure->CreateVertex(1.0f, 1.0f, 1.0f);
		structure->CreateVertex(1.0f, 1.0f, -1.0f);
		structure->CreateVertex(-1.0f, -1.0f, -1.0f);
		structure->CreateVertex(-1.0f, -1.0f, 1.0f);
		structure->CreateVertex(1.0f, -1.0f, 1.0f);
		structure->CreateVertex(1.0f, -1.0f, -1.0f);

		// texcrds
		structure->CreateTexcrd(0.0f, 0.0f);
		structure->CreateTexcrd(0.0f, 1.0f);
		structure->CreateTexcrd(1.0f, 1.0f);
		structure->CreateTexcrd(1.0f, 0.0f);

		// triangles
		// top
		structure->CreateTriangle({ 1, 2, 0 }, { 1, 2, 0 });
		structure->CreateTriangle({ 3, 0, 2 }, { 3, 0, 2 });
		// bottom
		structure->CreateTriangle({ 4, 7, 5 }, { 1, 2, 0 });
		structure->CreateTriangle({ 6, 5, 7 }, { 3, 0, 2 });
		// front
		structure->CreateTriangle({ 0, 3, 4 }, { 1, 2, 0 });
		structure->CreateTriangle({ 7, 4, 3 }, { 3, 0, 2 });
		// back
		structure->CreateTriangle({ 2, 1, 6 }, { 1, 2, 0 });
		structure->CreateTriangle({ 5, 6, 1 }, { 3, 0, 2 });
		// right
		structure->CreateTriangle({ 3, 2, 7 }, { 1, 2, 0 });
		structure->CreateTriangle({ 6, 7, 2 }, { 3, 0, 2 });
		// left
		structure->CreateTriangle({ 1, 0, 5 }, { 1, 2, 0 });
		structure->CreateTriangle({ 4, 5, 0 }, { 3, 0, 2 });

		conStruct.mesh_structure = structure;
		return world.Container<RZ::World::ContainerType::Mesh>().Create(conStruct);
	}

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