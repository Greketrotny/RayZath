#include "scene.h"
#include "application.h"
#include "loader.h"

#include <numeric>

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

			if (x == 0u || x == bitmap.GetWidth() - 1u || y == 0u || y == bitmap.GetHeight() - 1u)
			{
				bitmap.Value(x, y) = Graphics::Color::Palette::Yellow;
			}
		}
	}

	//bitmap.Value(0u, 0u) = Graphics::Color::Palette::Red;
	//bitmap.Value(bitmap.GetHeight() - 1u, 0u) = Graphics::Color::Palette::Green;
	//bitmap.Value(0u, bitmap.GetWidth() - 1u) = Graphics::Color::Palette::Blue;
	//bitmap.Value(bitmap.GetHeight() - 1u, bitmap.GetWidth() - 1u) = Graphics::Color::Palette::Black;

	return bitmap;
}

Graphics::Buffer2D<float> GenerateEmittanceMap(
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

Graphics::Buffer2D<uint8_t> GenerateReflectanceMap(
	const uint32_t resolution)
{
	Graphics::Buffer2D<uint8_t> reflectance_map(resolution, resolution);
	for (uint32_t x = 0u; x < resolution; x++)
	{
		for (uint32_t y = 0u; y < resolution; y++)
		{
			if ((x % 2 == 0) ^ (y % 2 == 0))
				reflectance_map.Value(x, y) = 0x0;
			else
				reflectance_map.Value(x, y) = 0xFF;
		}
	}

	return reflectance_map;
}

namespace Tester
{
	Scene::Scene(Application& app)
		: mr_app(app)
		, mr_engine(RZ::Engine::GetInstance())
		, mr_world(mr_engine.GetWorld())
	{
		// cameras
		m_camera = mr_world.Container<RZ::World::ContainerType::Camera>().Create(
			RZ::ConStruct<RZ::Camera>(
				L"camera 1",
				Math::vec3f(0.0f, 2.0f, -5.5f),
				Math::vec3f(0.0f, 0.0f, 0.0f),
				/*Math::vec3f(-2.0f, -4.0f, -14.0f),
				Math::vec3f(0.5f, -0.4f, 0.0f),*/
				1280u, 720u,
				Math::angle_degf(100.0f),
				5.5f, 0.001f, 0.016f, 0.75f, true));

		RZ::World& world = RZ::Engine::GetInstance().GetWorld();
		
		// lights
		RZ::Handle<RZ::PointLight> point_light1 = world.Container<RZ::World::ContainerType::PointLight>().Create(
			RZ::ConStruct<RZ::PointLight>(
				L"point light 1",
				Math::vec3f(2.0f, 3.0f, -2.0f),
				Graphics::Color::Palette::White,
				0.1f, 50.0f));
		/*world.Container<RZ::SpotLight>().Create(
			RZ::ConStruct<RZ::SpotLight>(
				L"spotlight 1",
				Math::vec3f(0.0f, 4.0f, -4.0f),
				Math::vec3f(0.0f, -1.0f, 1.0f),
				Graphics::Color::Palette::White,
				0.25f, 50.0f, 0.3f, 0.5f));*/
		/*mr_world.Container<RZ::World::ContainerType::DirectLight>().Create(
			RZ::ConStruct<RZ::DirectLight>(
				L"direct light 1",
				Math::vec3f(1.0f, -1.0f, 1.0f),
				Graphics::Color::Palette::White,
				10.0f, 0.02f));*/


		// textures
		RZ::Handle<RZ::Texture> tex_grid = world.Container<RZ::World::ContainerType::Texture>().Create(
			RZ::ConStruct<RZ::Texture>(
				L"texture grid",
				GenerateBitmap(
					128,
					Graphics::Color::Palette::White,
					Graphics::Color::Palette::Grey),
				RZ::Texture::FilterMode::Point,
				RZ::Texture::AddressMode::Wrap));

		RZ::Handle<RZ::Texture> tex_environment = world.Container<RZ::World::ContainerType::Texture>().Create(
			RZ::ConStruct<RZ::Texture>(
				L"texture 1",
				LoadFromFile(
					"D:/Users/Greketrotny/Documents/RayZath/Resources/img/environment.jpg"),
				RZ::Texture::FilterMode::Point));

		RZ::Handle<RZ::NormalMap> test_normal_map = world.Container<RZ::World::ContainerType::NormalMap>().Create(
			RZ::ConStruct<RZ::Texture>(
				L"test normal map",
				LoadFromFile(
					"D:/Users/Greketrotny/Documents/RayZath/Resources/img/rough_map.jpg"),
				RZ::Texture::FilterMode::Linear,
				RZ::Texture::AddressMode::Wrap));

		RZ::Handle<RZ::Texture> sphere_texture = world.Container<RZ::World::ContainerType::Texture>().Create(
			RZ::ConStruct<RZ::Texture>(
				L"sphere texture",
				LoadFromFile(
					"D:/Users/Greketrotny/Documents/RayZath/Resources/img/wood_color.jpg"),
				RZ::NormalMap::FilterMode::Linear));
		RZ::Handle<RZ::NormalMap> sphere_normal_map = world.Container<RZ::World::ContainerType::NormalMap>().Create(
			RZ::ConStruct<RZ::NormalMap>(
				L"sphere normal map",
				LoadFromFile(
					"D:/Users/Greketrotny/Documents/RayZath/Resources/img/TestNormalMap.jpg"),
				RZ::Texture::FilterMode::Linear));

		// emittance maps
		RZ::Handle<RZ::EmittanceMap> emittance_map = 
			world.Container<RZ::World::ContainerType::EmittanceMap>().Create(
			RZ::ConStruct<RZ::EmittanceMap>(
				L"emittance map",
				GenerateEmittanceMap(16u, 10.0f)));

		// reflectance maps
		RZ::Handle<RZ::ReflectanceMap> reflectance_map = 
			world.Container<RZ::World::ContainerType::ReflectanceMap>().Create(
			RZ::ConStruct < RZ::ReflectanceMap>(
				L"reflectance map",
				GenerateReflectanceMap(16u)));
		

		// world
		//world.GetMaterial().SetTexture(env_texture);
		//world.GetDefaultMaterial().SetColor(Graphics::Color::Palette::Green);
		//world.GetMaterial().SetEmittance(5.0f);
		//world.GetMaterial().SetScattering(0.05f);


		// materials
		RZ::Handle<RZ::Material> mat_ground = 
			world.Container<RZ::World::ContainerType::Material>().Create(
			RZ::ConStruct<RZ::Material>(
				Graphics::Color::Palette::DarkGreen,
				0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f,
				tex_grid));
		RZ::Handle<RZ::Material> sphere_material = world.Container<RZ::World::ContainerType::Material>().Create(
			RZ::ConStruct<RZ::Material>(
				Graphics::Color::Palette::Green,
				0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f));
		
		// spheres
		RZ::Handle<RZ::Sphere> sphere = world.Container<RZ::World::ContainerType::Sphere>().Create(
			RZ::ConStruct<RZ::Sphere>(
				L"sphere1",
				Math::vec3f(0.0f, 0.5f, 0.0f),
				Math::vec3f(0.0f),
				Math::vec3f(0.0f),
				Math::vec3f(0.5f),
				sphere_material));

		// cubes
		/*cube = CreateWoodenCrate(world, RZ::ConStruct<RZ::Mesh>(
			L"woonden crate",
			Math::vec3f(0.0f, 0.0f, 0.0f),
			Math::vec3f(0.0f),
			Math::vec3f(0.0f, 0.0f, 0.0f),
			Math::vec3f(0.2f)));*/

		/*CreateLightPlane(world, RZ::ConStruct<RZ::Mesh>(
			L"light plane",
			Math::vec3f(0.0f, 3.0f, 0.0f),
			Math::vec3f(0.0f),
			Math::vec3f(0.0f),
			Math::vec3f(1.0f)),
			Graphics::Color::Palette::White);*/



		// teapot
		//RZ::Handle<RZ::MeshStructure> teapot_structure = 
		//	world.Container<RZ::World::ContainerType::MeshStructure>().Create(
		//		RZ::ConStruct<RZ::MeshStructure>());
		//	teapot_structure->LoadFromFile(
		//		L"D:/Users/Greketrotny/Documents/RayZath/Resources/teapot.obj");
		//		//L"D:/Users/Greketrotny/Documents/Bleder Course/Level1/Part6/donut.obj");
		//	this->teapot = world.Container<RZ::World::ContainerType::Mesh>().Create(
		//		RZ::ConStruct<RZ::Mesh>(
		//			L"teapot",
		//			Math::vec3f(0.0f, 1.0f, -2.0f),
		//			Math::vec3f(0.0f, 1.57f, 0.0f),
		//			Math::vec3f(0.0f, 0.0f, 0.0f),
		//			Math::vec3f(1.0f, 1.0f, 1.0f),
		//			teapot_structure,
		//			mat_diffuse));

		RZ::Handle<RZ::Mesh> ground = CreateGround(mr_world, RZ::ConStruct<RZ::Mesh>(
			L"ground",
			Math::vec3f(0.0f, 0.0f, 0.0f),
			Math::vec3f(0.0f, 0.0f, 0.0f),
			Math::vec3f(0.0f, 0.0f, 0.0f),
			Math::vec3f(5.0f, 1.0f, 5.0f),
			RZ::Handle<RZ::MeshStructure>(),
			mat_ground));

		/*RZ::Handle<RZ::Plane> plane = world.Container<RZ::World::ContainerType::Plane>().Create(
			RZ::ConStruct<RZ::Plane>(L"plane1",
				Math::vec3f(0.0f),
				Math::vec3f(0.0f),
				Math::vec3f(0.0f),
				Math::vec3f(1.0f),
				mat_diffuse));*/
	}
	Scene::~Scene()
	{
	}

	void Scene::Render()
	{
		mr_engine.RenderWorld(RZ::Engine::RenderDevice::Default ,true, false);
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
		/*if (mr_world.GetStateRegister().RequiresUpdate())
		{
			m_camera->LookAtPoint(
				cube->GetTransformation().GetPosition(), 
				m_camera->GetRotation().z);
		}*/
		//bunny->LookAtPoint(m_camera->GetPosition() + m_camera->GetCoordSystem().GetZAxis() * 5.0f);

		const float d1 = m_camera->GetFocalDistance();

		const WAF::Point p = mr_app.m_ui.GetRenderWindow()->focal_point;
		const float d2 = m_camera->GetDepthBuffer().Value(p.x, p.y);
		if (mr_world.GetStateRegister().IsModified() || std::abs(d1 - d2) > 0.01f * d2)
		{
			m_camera->Focus(Math::vec2ui32(p.x, p.y));
		}

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
			RZ::ConStruct<RZ::MeshStructure>(8u, 4u, 0u, 12u));

		// vertices
		structure->CreateVertex(-1.0f, 1.0f, -1.0f);
		structure->CreateVertex(1.0f, 1.0f, -1.0f);
		structure->CreateVertex(1.0f, 1.0f, 1.0f);
		structure->CreateVertex(-1.0f, 1.0f, 1.0f);
		structure->CreateVertex(-1.0f, -1.0f, -1.0f);
		structure->CreateVertex(1.0f, -1.0f, -1.0f);
		structure->CreateVertex(1.0f, -1.0f, 1.0f);
		structure->CreateVertex(-1.0f, -1.0f, 1.0f);

		// texcrds
		structure->CreateTexcrd(0.0f, 0.0f);
		structure->CreateTexcrd(1.0f, 0.0f);
		structure->CreateTexcrd(0.0f, 1.0f);
		structure->CreateTexcrd(1.0f, 1.0f);

		// triangles
		auto& vertices = structure->GetVertices();
		auto& texcrds = structure->GetTexcrds();

		// front
		structure->CreateTriangle(
			&vertices[0], &vertices[1], &vertices[4],
			&texcrds[0], &texcrds[1], &texcrds[2]);
		structure->CreateTriangle(
			&vertices[5], &vertices[4], &vertices[1],
			&texcrds[3], &texcrds[2], &texcrds[1]);

		// right
		structure->CreateTriangle(
			&vertices[1], &vertices[2], &vertices[5],
			&texcrds[0], &texcrds[1], &texcrds[2]);
		structure->CreateTriangle(
			&vertices[6], &vertices[5], &vertices[2],
			&texcrds[3], &texcrds[2], &texcrds[1]);

		// back
		structure->CreateTriangle(
			&vertices[2], &vertices[3], &vertices[6],
			&texcrds[0], &texcrds[1], &texcrds[2]);
		structure->CreateTriangle(
			&vertices[7], &vertices[6], &vertices[3],
			&texcrds[3], &texcrds[2], &texcrds[1]);

		// left
		structure->CreateTriangle(
			&vertices[3], &vertices[0], &vertices[7],
			&texcrds[0], &texcrds[1], &texcrds[2]);
		structure->CreateTriangle(
			&vertices[4], &vertices[7], &vertices[0],
			&texcrds[3], &texcrds[2], &texcrds[1]);

		// top
		structure->CreateTriangle(
			&vertices[1], &vertices[0], &vertices[2],
			&texcrds[3], &texcrds[2], &texcrds[1]);
		structure->CreateTriangle(
			&vertices[3], &vertices[2], &vertices[0],
			&texcrds[0], &texcrds[1], &texcrds[2]);

		// bottom
		structure->CreateTriangle(
			&vertices[5], &vertices[6], &vertices[4],
			&texcrds[3], &texcrds[2], &texcrds[1]);
		structure->CreateTriangle(
			&vertices[7], &vertices[4], &vertices[6],
			&texcrds[0], &texcrds[1], &texcrds[2]);

		//RayZath::Texture t(GenerateBitmap(), RayZath::Texture::FilterMode::Point);
		//mesh->LoadTexture(t);

		conStruct.mesh_structure = structure;
		return world.Container<RZ::World::ContainerType::Mesh>().Create(conStruct);
	}
	RZ::Handle<RZ::Mesh> Scene::CreateRoom(
		RZ::World& world,
		RZ::ConStruct<RZ::Mesh> conStruct)
	{
		// [>] Create mesh structure
		conStruct.mesh_structure = world.Container<RZ::World::ContainerType::MeshStructure>().Create(
			RZ::ConStruct<RZ::MeshStructure>(8u, 14u, 18u, 16u));
		auto& structure = conStruct.mesh_structure;

		// vertices
		structure->CreateVertex(-1.0f, 1.0f, -1.0f);
		structure->CreateVertex(1.0f, 1.0f, -1.0f);
		structure->CreateVertex(1.0f, 1.0f, 1.0f);
		structure->CreateVertex(-1.0f, 1.0f, 1.0f);
		structure->CreateVertex(-1.0f, -1.0f, -1.0f);
		structure->CreateVertex(1.0f, -1.0f, -1.0f);
		structure->CreateVertex(1.0f, -1.0f, 1.0f);
		structure->CreateVertex(-1.0f, -1.0f, 1.0f);

		// texture coordinates
		//mesh->Texcrds.CreateTexcd(0.0f, 1.0f);
		//mesh->Texcrds.CreateTexcd(1.0f, 1.0f);
		//mesh->Texcrds.CreateTexcd(1.0f, 0.0f);
		//mesh->Texcrds.CreateTexcd(0.0f, 0.0f);
		// texture coordinates
		for (int i = 0; i <= 8; i++)
		{
			structure->CreateTexcrd(i / 8.0f, 0.0f);
			structure->CreateTexcrd(i / 8.0f, 1.0f);
		}

		//// texture bitmap
		//mesh->LoadTexture(RZ::Texture(GenerateBitmap(), RZ::Texture::FilterMode::Point));
		//mesh->LoadTexture(RZ::Texture(GenerateColorBitmap(), RZ::Texture::FilterMode::Point));
		/* main render: 66ms */

		//// [>] Creation and Description of each triangle
		///// floor
		//mesh->Triangles.CreateTriangle(vertices[4], vertices[7], vertices[6], mesh->Texcrds[0], mesh->Texcrds[3], mesh->Texcrds[2]);
		//mesh->Triangles.CreateTriangle(vertices[4], vertices[6], vertices[5], mesh->Texcrds[0], mesh->Texcrds[2], mesh->Texcrds[1]);
		///// ceil
		//mesh->Triangles.CreateTriangle(vertices[0], vertices[2], vertices[3], mesh->Texcrds[0], mesh->Texcrds[2], mesh->Texcrds[3]);
		//mesh->Triangles.CreateTriangle(vertices[0], vertices[1], vertices[2], mesh->Texcrds[0], mesh->Texcrds[1], mesh->Texcrds[2]);
		///// left wall
		//mesh->Triangles.CreateTriangle(vertices[0], vertices[3], vertices[7], mesh->Texcrds[0], mesh->Texcrds[3], mesh->Texcrds[2]);
		//mesh->Triangles.CreateTriangle(vertices[0], vertices[7], vertices[4], mesh->Texcrds[0], mesh->Texcrds[2], mesh->Texcrds[1]);
		///// right wall
		//mesh->Triangles.CreateTriangle(vertices[1], vertices[6], vertices[2], mesh->Texcrds[0], mesh->Texcrds[2], mesh->Texcrds[3]);
		//mesh->Triangles.CreateTriangle(vertices[1], vertices[5], vertices[6], mesh->Texcrds[0], mesh->Texcrds[1], mesh->Texcrds[2]);
		///// back wall
		//mesh->Triangles.CreateTriangle(vertices[3], vertices[2], vertices[6], mesh->Texcrds[0], mesh->Texcrds[3], mesh->Texcrds[2]);
		//mesh->Triangles.CreateTriangle(vertices[3], vertices[6], vertices[7], mesh->Texcrds[0], mesh->Texcrds[2], mesh->Texcrds[1]);
		///// front wall
		//mesh->Triangles.CreateTriangle(vertices[0], vertices[5], vertices[1], mesh->Texcrds[0], mesh->Texcrds[2], mesh->Texcrds[3]);
		//mesh->Triangles.CreateTriangle(vertices[0], vertices[4], vertices[5], mesh->Texcrds[0], mesh->Texcrds[1], mesh->Texcrds[2]);


		// [>] Creation and Description of each triangle
		auto& vertices = structure->GetVertices();
		auto& texcrds = structure->GetTexcrds();

		/// floor
		structure->CreateTriangle(
			&vertices[4], &vertices[7], &vertices[6], 
			&texcrds[1], &texcrds[0], &texcrds[2]);
		structure->CreateTriangle(
			&vertices[4], &vertices[6], &vertices[5],
			&texcrds[1], &texcrds[2], &texcrds[3]);
		//// ceil
		structure->CreateTriangle(
			&vertices[0], &vertices[2], &vertices[3],
			&texcrds[2], &texcrds[5], &texcrds[3]);
		structure->CreateTriangle(
			&vertices[0], &vertices[1], &vertices[2],
			&texcrds[2], &texcrds[4], &texcrds[5]);
		//// left wall
		structure->CreateTriangle(
			&vertices[0], &vertices[3], &vertices[7],
			&texcrds[4], &texcrds[6], &texcrds[7]);
		structure->CreateTriangle(
			&vertices[0], &vertices[7], &vertices[4],
			&texcrds[4], &texcrds[7], &texcrds[5]);
		//// right wall
		structure->CreateTriangle(
			&vertices[1], &vertices[6], &vertices[2],
			&texcrds[8], &texcrds[7], &texcrds[6]/*,
			nullptr, nullptr, nullptr,
			1u*/);
		structure->CreateTriangle(
			&vertices[1], &vertices[5], &vertices[6],
			&texcrds[8], &texcrds[9], &texcrds[7]/*,
			nullptr, nullptr, nullptr,
			1u*/);
		//// back wall
		structure->CreateTriangle(
			&vertices[3], &vertices[2], &vertices[6],
			&texcrds[8], &texcrds[10], &texcrds[11]);
		structure->CreateTriangle(
			&vertices[3], &vertices[6], &vertices[7],
			&texcrds[8], &texcrds[11], &texcrds[9]);
		/// front wall
		/*structure->CreateTriangle(
			&vertices[0], &vertices[5], &vertices[1],
			&texcrds[10], &texcrds[13], &texcrds[12]);
		structure->CreateTriangle(
			&vertices[0], &vertices[4], &vertices[5],
			&texcrds[10], &texcrds[11], &texcrds[13]);*/

		return world.Container<RZ::World::ContainerType::Mesh>().Create(conStruct);
	}


	RZ::Handle<RZ::Mesh> Scene::CreateGround(
		RZ::World& world,
		RZ::ConStruct<RZ::Mesh> construct)
	{
		construct.mesh_structure = world.Container<RZ::World::ContainerType::MeshStructure>().Create(
			RZ::ConStruct<RZ::MeshStructure>(4u, 4u, 4u, 2u));

		auto& structure = construct.mesh_structure;

		/*
		*	0 --------- 1
		*	| 0.1	1.1 |
		*	|			|
		*	|			|
		*	| 0.0   1.0	|
		*	3 --------- 2
		*/
		structure->CreateVertex(-1.0f, 0.0f, 1.0f);	// 0
		structure->CreateVertex(1.0f, 0.0f, 1.0f);	// 1
		structure->CreateVertex(1.0f, 0.0f, -1.0f);	// 2
		structure->CreateVertex(-1.0f, 0.0f, -1.0f);// 3

		structure->CreateTexcrd(0.0f, 1.0f);
		structure->CreateTexcrd(1.0f, 1.0f);
		structure->CreateTexcrd(1.0f, 0.0f);
		structure->CreateTexcrd(0.0f, 0.0f);

		structure->CreateTriangle(
			&structure->GetVertices()[0],
			&structure->GetVertices()[1],
			&structure->GetVertices()[2],
			&structure->GetTexcrds()[0],
			&structure->GetTexcrds()[1],
			&structure->GetTexcrds()[2]);
		structure->CreateTriangle(
			&structure->GetVertices()[0],
			&structure->GetVertices()[2],
			&structure->GetVertices()[3],
			&structure->GetTexcrds()[0],
			&structure->GetTexcrds()[2],
			&structure->GetTexcrds()[3]);

		return world.Container<RZ::World::ContainerType::Mesh>().Create(construct);
	}

	RZ::Handle<RZ::Mesh> Scene::CreateLightPlane(
		RZ::World& world,
		RZ::ConStruct<RZ::Mesh> con_struct,
		const Graphics::Color& color)
	{
		// mesh structure
		RZ::Handle<RZ::MeshStructure> structure = world.Container<RZ::World::ContainerType::MeshStructure>().Create(
			RZ::ConStruct<RZ::MeshStructure>(4u, 0u, 0u, 2u));

		structure->CreateVertex(-1.0f, 0.0f, -1.0f);
		structure->CreateVertex(1.0f, 0.0f, -1.0f);
		structure->CreateVertex(1.0f, 0.0f, 1.0f);
		structure->CreateVertex(-1.0f, 0.0f, 1.0f);

		structure->CreateTriangle(
			&structure->GetVertices()[0],
			&structure->GetVertices()[1],
			&structure->GetVertices()[2]);
		structure->CreateTriangle(
			&structure->GetVertices()[0],
			&structure->GetVertices()[2],
			&structure->GetVertices()[3]);

		// material
		RZ::Handle<RZ::Material> material = world.Container<RZ::World::ContainerType::Material>().Create(
			RZ::ConStruct<RZ::Material>(
				color,
				1.0f, 0.0f, 0.0f, 1.0f, 50.0f, 0.0f));

		con_struct.material = material;
		con_struct.mesh_structure = structure;

		return world.Container<RZ::World::ContainerType::Mesh>().Create(con_struct);
	}

	RZ::Handle<RZ::Mesh> Scene::CreateRoundedCube(
		RZ::World& world,
		RZ::ConStruct<RZ::Mesh> con_struct)
	{
		// mesh structure
		con_struct.mesh_structure = world.Container<RZ::World::ContainerType::MeshStructure>().Create(
			RZ::ConStruct<RZ::MeshStructure>());
		con_struct.mesh_structure->LoadFromFile(
			L"D:/Users/Greketrotny/Documents/RayZath/Resources/rounded-cube.obj");

		return world.Container<RZ::World::ContainerType::Mesh>().Create(con_struct);
	}

	RZ::Handle<RZ::Mesh> Scene::CreateWoodenCrate(
		RZ::World& world,
		RZ::ConStruct<RZ::Mesh> con_struct)
	{
		// mesh data
		con_struct.mesh_structure = world.Container<RZ::World::ContainerType::MeshStructure>().Create(
			RZ::ConStruct<RZ::MeshStructure>());
		con_struct.mesh_structure->LoadFromFile(
			L"D:/Users/Greketrotny/Documents/RayZath/Resources/wooden_crate/Wooden Crate.obj");

		// textures
		RZ::Handle<RZ::Texture> texture = world.Container<RZ::World::ContainerType::Texture>().Create(
			RZ::ConStruct<RZ::Texture>(
				L"crate_texture",
				LoadFromFile(
					"D:/Users/Greketrotny/Documents/RayZath/Resources/wooden_crate/Textures/1024/wooden_crate_texture.jpg"),
				RZ::Texture::FilterMode::Linear));

		// normal map
		RZ::Handle<RZ::NormalMap> normal_map = world.Container<RZ::World::ContainerType::NormalMap>().Create(
			RZ::ConStruct<RZ::NormalMap>(
				L"crate_normal_map",
				LoadFromFile(
					"D:/Users/Greketrotny/Documents/RayZath/Resources/wooden_crate/Textures/1024/wooden_crate_normal_map.jpg"),
				RZ::Texture::FilterMode::Linear));

		// material
		con_struct.material = world.Container<RZ::World::ContainerType::Material>().Create(
			RZ::ConStruct<RZ::Material>(
				Graphics::Color::Palette::Brown,
				0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f,
				texture, normal_map));

		return world.Container<RZ::World::ContainerType::Mesh>().Create(con_struct);
	}
	
	//void Scene::CreateTessellatedSphere(
	//	RZ::World* world,
	//	const RZ::ConStruct<RZ::Mesh>& conStruct,
	//	const uint32_t& res)
	//{
	//	// [>] create Mesh object
	//	RZ::Mesh* mesh = world->GetMeshes().Create(conStruct);
	//	mesh->GetMeshStructure()->Reset(10000u, 2u, 10000u, 10000u);

	//	// [>] Create vertices
	//	// middle layer vertices
	//	for (uint32_t i = 1; i < res - 1; ++i)
	//	{
	//		for (uint32_t j = 0; j < res; ++j)
	//		{
	//			RZ::Vertex v(0.0f, 1.0f, 0.0f);
	//			v.RotateX(((float)i / (float)(res - 1)) * Math::constants<float>::pi);
	//			v.RotateY(((float)j / (float)(res)) * Math::constants<float>::pi * 2.0f);
	//			mesh->GetMeshStructure()->CreateVertex(v);
	//			mesh->GetMeshStructure()->CreateNormal(v);
	//		}
	//	}

	//	// top and bottom vertices
	//	Math::vec3f* vTop = mesh->GetMeshStructure()->CreateVertex(0.0f, 1.0f, 0.0f);
	//	Math::vec3f* vBottom = mesh->GetMeshStructure()->CreateVertex(0.0f, -1.0f, 0.0f);

	//	Math::vec3f* vNTop = mesh->GetMeshStructure()->CreateNormal(0.0f, 1.0f, 0.0f);
	//	Math::vec3f* vNBottom = mesh->GetMeshStructure()->CreateNormal(0.0f, -1.0f, 0.0f);


	//	// [>] Create triangles
	//	// hat and foot
	//	for (uint32_t i = 0; i < res; ++i)
	//	{
	//		mesh->GetMeshStructure()->CreateTriangle(
	//			vTop, 
	//			&mesh->GetMeshStructure()->GetVertices()[(i + 1) % res],
	//			&mesh->GetMeshStructure()->GetVertices()[i],
	//			nullptr, nullptr, nullptr,
	//			vNTop,
	//			&mesh->GetMeshStructure()->GetNormals()[(i + 1) % res],
	//			&mesh->GetMeshStructure()->GetNormals()[i]);
	//		mesh->GetMeshStructure()->CreateTriangle(
	//			vBottom,
	//			&mesh->GetMeshStructure()->GetVertices()[res * (res - 3) + i % res],
	//			&mesh->GetMeshStructure()->GetVertices()[res * (res - 3) + (i + 1) % res],
	//			nullptr, nullptr, nullptr,
	//			vNBottom,
	//			&mesh->GetMeshStructure()->GetNormals()[res * (res - 3) + i % res],
	//			&mesh->GetMeshStructure()->GetNormals()[res * (res - 3) + (i + 1) % res]);
	//	}

	//	// middle layers
	//	for (uint32_t i = 0; i < res - 3; ++i)
	//	{
	//		for (uint32_t j = 0; j < res; ++j)
	//		{
	//			mesh->GetMeshStructure()->CreateTriangle(
	//				&mesh->GetMeshStructure()->GetVertices()[i * res + j], 
	//				&mesh->GetMeshStructure()->GetVertices()[(i + 1) * res + (j + 1) % res], 
	//				&mesh->GetMeshStructure()->GetVertices()[(i + 1) * res + j],
	//				nullptr, nullptr, nullptr,
	//				&mesh->GetMeshStructure()->GetNormals()[i * res + j],
	//				&mesh->GetMeshStructure()->GetNormals()[(i + 1) * res + (j + 1) % res],
	//				&mesh->GetMeshStructure()->GetNormals()[(i + 1) * res + j]);

	//			mesh->GetMeshStructure()->CreateTriangle(
	//				&mesh->GetMeshStructure()->GetVertices()[i * res + j], 
	//				&mesh->GetMeshStructure()->GetVertices()[i * res + (j + 1) % res], 
	//				&mesh->GetMeshStructure()->GetVertices()[(i + 1) * res + (j + 1) % res],
	//				nullptr, nullptr, nullptr,
	//				&mesh->GetMeshStructure()->GetNormals()[i * res + j],
	//				&mesh->GetMeshStructure()->GetNormals()[i * res + (j + 1) % res],
	//				&mesh->GetMeshStructure()->GetNormals()[(i + 1) * res + (j + 1) % res]);
	//		}
	//	}


	//	// triangle coloring
	//	for (uint32_t i = 0; i < mesh->GetMeshStructure()->GetTriangles().GetCapacity(); ++i)
	//	{
	//		mesh->GetMeshStructure()->GetTriangles()[i].color = 
	//			Graphics::Color(
	//				rand() % 63 + 192,
	//				rand() % 63 + 192, 
	//				rand() % 63 + 192,
	//				0x00);
	//	}
	//}
}