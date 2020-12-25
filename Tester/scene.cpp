#include "scene.h"
#include "application.h"

Graphics::Bitmap GenerateColorBitmap()
{
	unsigned int resolution = 32;
	Graphics::Bitmap bitmap(resolution * 8, resolution);

	unsigned char hs = 0xFF;
	unsigned char ls = 0x22;
	std::vector<Graphics::Color> colors{
		Graphics::Color(hs, ls, ls),	// red
		Graphics::Color(ls, hs, ls),	// green
		Graphics::Color(ls, ls, hs),	// blue
		Graphics::Color(ls, hs, hs),	// cyan
		Graphics::Color(hs, hs, ls),	// yellow
		Graphics::Color(hs, ls, hs),	// magenta
		Graphics::Color(hs, hs, hs),	// white
		Graphics::Color(ls, ls, ls),	// dark grey
	};
	//std::vector<Graphics::Color> colors{
	//	Graphics::Color(hs, hs, hs),	// white
	//	Graphics::Color(ls, hs, ls),	// green
	//	Graphics::Color(hs, hs, hs),	// white
	//	Graphics::Color(hs, hs, hs),	// white
	//	Graphics::Color(ls, ls, hs),	// white
	//	Graphics::Color(ls, ls, hs),	// white
	//	Graphics::Color(ls, ls, hs),	// white
	//	Graphics::Color(ls, ls, ls),	// dark grey
	//};

	for (unsigned int i = 0; i < 8; ++i)
	{
		for (unsigned int x = 0; x < resolution; ++x)
		{
			for (unsigned int y = 0; y < resolution; ++y)
			{
				if ((x % 2 == 0) ^ (y % 2 == 0)) bitmap.SetPixel(resolution * i + x, y, colors[i]);
				else bitmap.SetPixel(resolution * i + x, y, Graphics::Color::BlendAverage(colors[i], Graphics::Color(0x00, 0x00, 0x00)));

				//if (x == 2 && y == 2)
				//	bitmap.SetPixel(x, y, Graphics::Color(0x00, 0x00, 0xFF));
			}
		}

		/*if (i == 1)
		{
			bitmap.SetPixel(resolution * i + resolution / 2, resolution / 2, Graphics::Color(0xFF, 0xFF, 0xFF, 0x00));
			bitmap.SetPixel(resolution * i + resolution / 2 + 1, resolution / 2, Graphics::Color(0xFF, 0xFF, 0xFF, 0x00));
			bitmap.SetPixel(resolution * i + resolution / 2 + 1, resolution / 2 + 1, Graphics::Color(0xFF, 0xFF, 0xFF, 0x00));
			bitmap.SetPixel(resolution * i + resolution / 2, resolution / 2 + 1, Graphics::Color(0xFF, 0xFF, 0xFF, 0x00));
		}*/
	}
	
	return bitmap;
}

Graphics::Bitmap GenerateBitmap(
	const uint32_t resolution, 
	const Graphics::Color& color1,
	const Graphics::Color& color2)
{
	Graphics::Bitmap bitmap(resolution, resolution);

	for (unsigned int x = 0; x < resolution; ++x)
	{
		for (unsigned int y = 0; y < resolution; ++y)
		{
			if ((x % 2 == 0) ^ (y % 2 == 0)) 
				bitmap.SetPixel(x, y, color1);
			else 
				bitmap.SetPixel(x, y, color2);
		}
	}
	return bitmap;
}


namespace Tester
{
	Scene::Scene(Application& app)
		: mr_app(app)
		, mr_engine(RZ::Engine::GetInstance())
		, mr_world(mr_engine.GetWorld())
	{
		// cameras
		m_camera = mr_world.GetCameras().CreateObject(
			RZ::ConStruct<RZ::Camera>(
				L"camera 1",
				Math::vec3<float>(0.0f, 3.0f, -11.0f),
				Math::vec3<float>(0.0f, 0.0f, 0.0f),
				1200u, 700u,
				Math::angle_degf(100.0f),
				10.0f, 0.001f, true));

		RZ::World& world = RZ::Engine::GetInstance().GetWorld();

		// lights
		/*RZ::Handle<RZ::PointLight> point_light1 = world.GetPointLights().CreateObject(
			RZ::ConStruct<RZ::PointLight>(
				L"point light 1",
				Math::vec3f(0.0f, 3.0f, 0.0f),
				Graphics::Color::White,
				0.5f, 10.0f));*/
		/*world.GetSpotLights().CreateObject(
			RZ::ConStruct<RZ::SpotLight>(
				L"spotlight 1",
				Math::vec3<float>(0.0f, 4.0f, -4.0f),
				Math::vec3<float>(0.0f, -1.0f, 1.0f),
				Graphics::Color(0xFF, 0xFF, 0xFF),
				0.25f, 50.0f, 0.3f, 2.0f));*/
		/*mr_world.GetDirectLights().CreateObject(
			RZ::ConStruct<RZ::DirectLight>(
				L"direct light 1",
				Math::vec3<float>(1.0f, -1.0f, 1.0f),
				Graphics::Color(0xFF, 0xFF, 0xFF, 0xFF),
				10.0f, 0.05f));*/

		// textures
		RZ::Handle<RZ::Texture> texture1 = world.GetTextures().CreateObject(
			RZ::ConStruct<RZ::Texture>(
				L"texture 1",
				GenerateBitmap(
					8u,
					Graphics::Color(0xFF, 0xFF, 0xFF, 0x00),
					Graphics::Color(0x80, 0x80, 0x80, 0x00)),
				RZ::Texture::FilterMode::Point));
		RZ::Handle<RZ::Texture> texture2 = world.GetTextures().CreateObject(
			RZ::ConStruct<RZ::Texture>(
				L"texture 2",
				GenerateColorBitmap(),
				RZ::Texture::FilterMode::Point));

		// materials
		RZ::Handle<RZ::Material> mat_diffuse = world.GetMaterials().CreateObject(
			RZ::ConStruct<RZ::Material>(
				Graphics::Color(0xC0, 0xC0, 0xC0, 0x00),
				0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f,
				texture1));
		RZ::Handle<RZ::Material> mat_diffuse2 = world.GetMaterials().CreateObject(
			RZ::ConStruct<RZ::Material>(
				Graphics::Color(0xC0, 0xC0, 0xC0, 0x00),
				0.0f, 0.0f, 0.0f, 1.5f, 0.0f, 0.0f,
				/*RZ::Handle<RZ::Texture>()*/texture2));
		RZ::Handle<RZ::Material> mat_glass = world.GetMaterials().CreateObject(
			RZ::ConStruct<RZ::Material>(
				Graphics::Color(0xFF, 0xFF, 0xFF, 0x00),
				0.0f, 0.0f, 1.0f, 1.5f, 0.0f, 0.0f));
		RZ::Handle<RZ::Material> mat_mirror = world.GetMaterials().CreateObject(
			RZ::ConStruct<RZ::Material>(
				Graphics::Color(0xFF, 0xFF, 0xFF, 0x00),
				0.8f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f));

		// spheres
		RZ::Handle<RZ::Sphere> sphere = world.GetSpheres().CreateObject(
			RZ::ConStruct<RZ::Sphere>(
				L"glass sphere",
				Math::vec3<float>(2.0f, 3.0f, -0.5f),
				Math::vec3<float>(0.0f, 0.0f, 0.0f),
				Math::vec3<float>(0.0f, 0.0f, 0.0f),
				Math::vec3<float>(1.0f, 1.0f, 1.0f),
				mat_glass));
		
		// cubes
		CreateCube(world, RZ::ConStruct<RZ::Mesh>(
			L"tall cube",
			Math::vec3<float>(-2.0f, 0.0f, 1.0f),
			Math::vec3<float>(
				0.0f,
				Math::angle_radf(Math::angle_degf(35.0f)).value(),
				0.0f),
			Math::vec3<float>(0.0f, 1.0f, 0.0f),
			Math::vec3<float>(1.0f, 2.0f, 1.0f),	
			mat_mirror));
		CreateCube(world, RZ::ConStruct<RZ::Mesh>(
			L"front cube",
			Math::vec3<float>(2.0f, 0.0f, -0.5f),
			Math::vec3<float>(
				0.0f,
				Math::angle_radf(Math::angle_degf(-25.0f)).value(),
				0.0f),
			Math::vec3<float>(0.0f, 1.0f, 0.0f),
			Math::vec3<float>(1.0f, 1.0f, 1.0f),
			mat_diffuse));

		// light planes
		CreateLightPlane(
			world,
			RZ::ConStruct<RZ::Mesh>(
				L"light plane",
				Math::vec3<float>(0.0f, 5.99f, 0.0f),
				Math::vec3<float>(0.0f, 0.0f, 0.0f),
				Math::vec3<float>(0.0f, 0.0f, 0.0f),
				Math::vec3<float>(1.0f, 1.0f, 1.0f)),
			Graphics::Color(0xFF, 0xFF, 0xFF));


		CreateRoom(mr_world, RZ::ConStruct<RZ::RenderObject>(
			L"Room",
			Math::vec3<float>(0.0f, 0.0f, 0.0f),
			Math::vec3<float>(0.0f, 0.0f, 0.0f),
			Math::vec3<float>(0.0f, 1.0f, 0.0f),
			Math::vec3<float>(5.0f, 3.0f, 3.0f),
			mat_diffuse2));
	}
	/*Scene::Scene(Application& app)
		: mr_app(app)
		, mr_engine(RZ::Engine::GetInstance())
		, mr_world(mr_engine.GetWorld())
	{
		// cameras
		m_camera = mr_world.GetCameras().CreateObject(RZ::ConStruct<RZ::Camera>(
			RZ::ConStruct<RZ::WorldObject>(L"camera 1"),
			Math::vec3<float>(0.0f, 3.0f, -11.0f),
			Math::vec3<float>(0.0f, 0.0f, 0.0f),
			1200, 700,
			Math::angle_degf(100.0),
			10.0f, 0.000f, true));

		RZ::World& world = RZ::Engine::GetInstance().GetWorld();

		// materials
		RZ::Material* glass = world.GetMaterials().CreateObject(
			RZ::ConStruct<RZ::Material>(
				Graphics::Color(0xFF, 0xFF, 0xFF, 0x00),
				0.0f, 0.0f, 1.0f, 1.5f, 0.0f, 0.0f));
		RZ::Material* mirror = world.GetMaterials().CreateObject(
			RZ::ConStruct<RZ::Material>(
				Graphics::Color(0xFF, 0xFF, 0xFF, 0x00),
				1.0f, 0.0f, 0.0f, 1.5f, 0.0f, 0.0f));
		RZ::Material* diffuse = world.GetMaterials().CreateObject(
			RZ::ConStruct<RZ::Material>(
				Graphics::Color(0xFF, 0xFF, 0xFF, 0x00),
				0.0f, 0.0f, 0.0f, 1.5f, 0.0f, 0.0f));
		RZ::Material* mat_light = world.GetMaterials().CreateObject(
			RZ::ConStruct<RZ::Material>(
				Graphics::Color(0xFF, 0xFF, 0xFF, 0x00),
				0.0f, 0.0f, 0.0f, 1.5f, 50.0f, 0.0f));

		// spheres
		RZ::Sphere* sphere = mr_world.GetSpheres().CreateObject(RZ::ConStruct<RZ::Sphere>(
			RZ::ConStruct<RZ::RenderObject>(
				RZ::ConStruct<RZ::WorldObject>(L"glass sphere"),
				Math::vec3<float>(2.0f, 3.0f, -0.5f),
				Math::vec3<float>(0.0f, 0.0f, 0.0f),
				Math::vec3<float>(0.0f, 0.0f, 0.0f),
				Math::vec3<float>(1.0f, 1.0f, 1.0f),
				glass),
			1.0f));


		CreateRoom(mr_world, RZ::ConStruct<RZ::RenderObject>(
				RZ::ConStruct<RZ::WorldObject>(L"Room"),
				Math::vec3<float>(0.0f, 0.0f, 0.0f),
				Math::vec3<float>(0.0f, 0.0f, 0.0f),
				Math::vec3<float>(0.0f, 1.0f, 0.0f),
				Math::vec3<float>(5.0f, 3.0f, 3.0f),
				diffuse));

		CreateCube(&mr_world, RZ::ConStruct<RZ::Mesh>(
			RZ::ConStruct<RZ::RenderObject>(
				RZ::ConStruct<RZ::WorldObject>(
					L"tall cube"),
				Math::vec3<float>(-2.0f, 0.0f, 1.0f),
				Math::vec3<float>(
					0.0f,
					Math::angle_radf(Math::angle_degf(35.0f)).value(),
					0.0f),
				Math::vec3<float>(0.0f, 1.0f, 0.0f),
				Math::vec3<float>(1.0f, 2.0f, 1.0f),
				mirror)));
		RZ::Mesh* front_cube = CreateCube(&mr_world, RZ::ConStruct<RZ::Mesh>(
			RZ::ConStruct<RZ::RenderObject>(
				RZ::ConStruct<RZ::WorldObject>(
					L"front cube"),
				Math::vec3<float>(2.0f, 0.0f, -0.5f),
				Math::vec3<float>(
					0.0f, 
					Math::angle_radf(Math::angle_degf(-25.0f)).value(),
					0.0f),
				Math::vec3<float>(0.0f, 1.0f, 0.0f),
				Math::vec3<float>(1.0f, 1.0f, 1.0f),
				diffuse)));
		front_cube->LoadTexture(RZ::Texture(GenerateBitmap(), RZ::Texture::FilterMode::Point));

		CreateLightPlane(
			world,
			RZ::ConStruct<RZ::Mesh>(
				RZ::ConStruct<RZ::RenderObject>(
					RZ::ConStruct<RZ::WorldObject>(
						L"light plane"),
					Math::vec3<float>(0.0f, 5.99f, 0.0f),
					Math::vec3<float>(0.0f, 0.0f, 0.0f),
					Math::vec3<float>(0.0f, 0.0f, 0.0f),
					Math::vec3<float>(1.0f, 1.0f, 1.0f),
					mat_light)),
			Graphics::Color(0xFF, 0xFF, 0xFF));
	}*/
	Scene::~Scene()
	{
	}

	void Scene::Render()
	{
		mr_engine.RenderWorld();
	}
	const Graphics::Bitmap& Scene::GetRender()
	{
		return m_camera->GetBitmap();
	}
	void Scene::ResizeRender(uint32_t width, uint32_t height)
	{
		m_camera->Resize(width, height);
	}
	
	RZ::Handle<RZ::Mesh> Scene::CreateCube(
		RZ::World& world, 
		RZ::ConStruct<RZ::Mesh> conStruct)
	{
		// create mesh structure
		RZ::Handle<RZ::MeshStructure> structure = world.GetMeshStructures().CreateObject(
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
			&texcrds[0], &texcrds[2], &texcrds[1]);
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


		// triangles colors
		unsigned char lc = 0x44;
		unsigned char hc = 0xFF;
		std::vector<Graphics::Color> colors{
			Graphics::Color(hc, lc, lc),
			Graphics::Color(lc, hc, lc),
			Graphics::Color(lc, lc, hc),
			Graphics::Color(hc, hc, lc),
			Graphics::Color(hc, lc, hc),
			Graphics::Color(lc, hc, hc) };

		auto& triangles = structure->GetTriangles();
		for (unsigned int i = 0; i < triangles.GetCount() / 2; i++)
		{
			//mesh->Triangles[2 * i]->Color(colors[i]);
			//mesh->Triangles[2 * i + 1]->Color(colors[i]);
			//triangles[2 * i].color = Graphics::Color(0xC0, 0xC0, 0xC0, 0x00);
			//triangles[2 * i + 1].color = Graphics::Color(0xC0, 0xC0, 0xC0, 0x00);
			triangles[2 * i].color = Graphics::Color(0xFF, 0xFF, 0xFF, 0x00);
			triangles[2 * i + 1].color = Graphics::Color(0xFF, 0xFF, 0xFF, 0x00);
		}

		//RayZath::Texture t(GenerateBitmap(), RayZath::Texture::FilterMode::Point);
		//mesh->LoadTexture(t);

		conStruct.mesh_structure = structure;
		return world.GetMeshes().CreateObject(conStruct);
	}
	void Scene::CreateRoom(
		RZ::World& world,
		const RZ::ConStruct<RZ::RenderObject>& conStruct)
	{
		// [>] Create mesh structure
		RZ::Handle<RZ::MeshStructure> structure = world.GetMeshStructures().CreateObject(
			RZ::ConStruct<RZ::MeshStructure>(8u, 14u, 18u, 16u));

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
			&texcrds[8], &texcrds[7], &texcrds[6]);
		structure->CreateTriangle(
			&vertices[1], &vertices[5], &vertices[6],
			&texcrds[8], &texcrds[9], &texcrds[7]);
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

		using namespace Graphics;
		//// floor
		structure->GetTriangles()[0].color = Color(0xC0, 0xC0, 0xC0, 0x00);
		structure->GetTriangles()[1].color = Color(0xC0, 0xC0, 0xC0, 0x00);
		//// ceil
		structure->GetTriangles()[2].color = Color(0xC0, 0xC0, 0xC0, 0x00);
		structure->GetTriangles()[3].color = Color(0xC0, 0xC0, 0xC0, 0x00);
		//// left wall
		structure->GetTriangles()[4].color = Color(0xC0, 0x40, 0x40, 0x00);
		structure->GetTriangles()[5].color = Color(0xC0, 0x40, 0x40, 0x00);
		//// right wall
		structure->GetTriangles()[6].color = Color(0x40, 0xC0, 0x40, 0x00);
		structure->GetTriangles()[7].color = Color(0x40, 0xC0, 0x40, 0x00);
		//// back wall
		structure->GetTriangles()[8].color = Color(0xC0, 0xC0, 0xC0, 0x00);
		structure->GetTriangles()[9].color = Color(0xC0, 0xC0, 0xC0, 0x00);
		//// front wall
		structure->GetTriangles()[8].color = Color(0xC0, 0xC0, 0xC0, 0x00);
		structure->GetTriangles()[9].color = Color(0xC0, 0xC0, 0xC0, 0x00);

		world.GetMeshes().CreateObject(
			RZ::ConStruct<RZ::Mesh>(
				conStruct.name,
				conStruct.position,
				conStruct.rotation,
				conStruct.center,
				conStruct.scale,
				conStruct.material,
				structure));
	}

	RZ::Handle<RZ::Mesh> Scene::CreateLightPlane(
		RZ::World& world,
		RZ::ConStruct<RZ::Mesh> con_struct,
		const Graphics::Color& color)
	{
		// mesh structure
		RZ::Handle<RZ::MeshStructure> structure = world.GetMeshStructures().CreateObject(
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

		structure->GetTriangles()[0].color = color;
		structure->GetTriangles()[1].color = color;

		// material
		RZ::Handle<RZ::Material> material = world.GetMaterials().CreateObject(
			RZ::ConStruct<RZ::Material>(
				color,
				1.0f, 0.0f, 0.0f, 1.0f, 50.0f, 0.0f));

		con_struct.material = material;
		con_struct.mesh_structure = structure;

		return world.GetMeshes().CreateObject(con_struct);
	}
	//void Scene::CreateTessellatedSphere(
	//	RZ::World* world,
	//	const RZ::ConStruct<RZ::Mesh>& conStruct,
	//	const uint32_t& res)
	//{
	//	// [>] create Mesh object
	//	RZ::Mesh* mesh = world->GetMeshes().CreateObject(conStruct);
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
	//	Math::vec3<float>* vTop = mesh->GetMeshStructure()->CreateVertex(0.0f, 1.0f, 0.0f);
	//	Math::vec3<float>* vBottom = mesh->GetMeshStructure()->CreateVertex(0.0f, -1.0f, 0.0f);

	//	Math::vec3<float>* vNTop = mesh->GetMeshStructure()->CreateNormal(0.0f, 1.0f, 0.0f);
	//	Math::vec3<float>* vNBottom = mesh->GetMeshStructure()->CreateNormal(0.0f, -1.0f, 0.0f);


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

	//RZ::Mesh* Scene::CreateRoundedCube(
	//	RZ::World& world,
	//	const RZ::ConStruct<RZ::Mesh>& con_struct)
	//{
	//	RZ::Mesh* mesh = world.GetMeshes().CreateObject(con_struct);
	//	if (mesh == nullptr) return nullptr;

	//	mesh->GetMeshStructure()->LoadFromFile(
	//		L"D:/Users/Greketrotny/Programming/Projects/C++/RayZath/Tester/Resources/rounded-cube.obj");

	//	//mesh->LoadTexture(RZ::Texture(GenerateBitmap(), RZ::Texture::FilterMode::Point));

	//	for (uint32_t i = 0u; i < mesh->GetMeshStructure()->GetTriangles().GetCount(); i++)
	//	{
	//		mesh->GetMeshStructure()->GetTriangles()[i].color = Graphics::Color(0xFF, 0xFF, 0xFF, 0x00);
	//	}
	//	return mesh;

	//	const float r = 0.9f;

	//	// [>] vertices
	//	// top
	//	mesh->GetMeshStructure()->CreateVertex(-1.0f * r, 1.0f, -1.0f * r);
	//	mesh->GetMeshStructure()->CreateVertex(1.0f * r, 1.0f, -1.0f * r);
	//	mesh->GetMeshStructure()->CreateVertex(1.0f * r, 1.0f, 1.0f * r);
	//	mesh->GetMeshStructure()->CreateVertex(-1.0f * r, 1.0f, 1.0f * r);

	//	// bottom
	//	mesh->GetMeshStructure()->CreateVertex(-1.0f * r, -1.0f, -1.0f * r);
	//	mesh->GetMeshStructure()->CreateVertex(1.0f * r, -1.0f, -1.0f * r);
	//	mesh->GetMeshStructure()->CreateVertex(1.0f * r, -1.0f, 1.0f * r);
	//	mesh->GetMeshStructure()->CreateVertex(-1.0f * r, -1.0f, 1.0f * r);

	//	// left
	//	mesh->GetMeshStructure()->CreateVertex(-1.0f, 1.0f * r, -1.0f * r);
	//	mesh->GetMeshStructure()->CreateVertex(-1.0f, -1.0f * r, -1.0f * r);
	//	mesh->GetMeshStructure()->CreateVertex(-1.0f, -1.0f * r, 1.0f * r);
	//	mesh->GetMeshStructure()->CreateVertex(-1.0f, 1.0f * r, 1.0f * r);

	//	// right
	//	mesh->GetMeshStructure()->CreateVertex(1.0f, 1.0f * r, -1.0f * r);
	//	mesh->GetMeshStructure()->CreateVertex(1.0f, -1.0f * r, -1.0f * r);
	//	mesh->GetMeshStructure()->CreateVertex(1.0f, -1.0f * r, 1.0f * r);
	//	mesh->GetMeshStructure()->CreateVertex(1.0f, 1.0f * r, 1.0f * r);

	//	// back
	//	mesh->GetMeshStructure()->CreateVertex(-1.0f * r, 1.0f * r, 1.0f);
	//	mesh->GetMeshStructure()->CreateVertex(-1.0f * r, -1.0f * r, 1.0f);
	//	mesh->GetMeshStructure()->CreateVertex(1.0f * r, -1.0f * r, 1.0f);
	//	mesh->GetMeshStructure()->CreateVertex(1.0f * r, 1.0f * r, 1.0f);

	//	// front
	//	mesh->GetMeshStructure()->CreateVertex(-1.0f * r, 1.0f * r, -1.0f);
	//	mesh->GetMeshStructure()->CreateVertex(1.0f * r, 1.0f * r, -1.0f);
	//	mesh->GetMeshStructure()->CreateVertex(1.0f * r, -1.0f * r, -1.0f);
	//	mesh->GetMeshStructure()->CreateVertex(-1.0f * r, -1.0f * r, -1.0f);


	//	// [>] Normals
	//	mesh->GetMeshStructure()->CreateNormal(0.0f, 1.0f, 0.0f);	// top		0
	//	mesh->GetMeshStructure()->CreateNormal(0.0f, -1.0f, 0.0f);	// bottom	1
	//	mesh->GetMeshStructure()->CreateNormal(-1.0f, 0.0f, 0.0f);	// left		2
	//	mesh->GetMeshStructure()->CreateNormal(1.0f, 0.0f, 0.0f);	// right	3
	//	mesh->GetMeshStructure()->CreateNormal(0.0f, 0.0f, 1.0f);	// back		4
	//	mesh->GetMeshStructure()->CreateNormal(0.0f, 0.0f, -1.0f);	// front	5


	//	// [>] Triangles
	//	auto& vertices = mesh->GetMeshStructure()->GetVertices();
	//	auto& normals = mesh->GetMeshStructure()->GetNormals();

	//	// faces
	//	// top
	//	mesh->GetMeshStructure()->CreateTriangle(
	//		&vertices[0], &vertices[3], &vertices[2],
	//		nullptr, nullptr, nullptr,
	//		&normals[0], &normals[0], &normals[0]);
	//	mesh->GetMeshStructure()->CreateTriangle(
	//		&vertices[0], &vertices[2], &vertices[1],
	//		nullptr, nullptr, nullptr,
	//		&normals[0], &normals[0], &normals[0]);

	//	// bottom
	//	mesh->GetMeshStructure()->CreateTriangle(
	//		&vertices[4], &vertices[7], &vertices[6],
	//		nullptr, nullptr, nullptr,
	//		&normals[1], &normals[1], &normals[1]);
	//	mesh->GetMeshStructure()->CreateTriangle(
	//		&vertices[4], &vertices[6], &vertices[5],
	//		nullptr, nullptr, nullptr,
	//		&normals[1], &normals[1], &normals[1]);

	//	// left
	//	mesh->GetMeshStructure()->CreateTriangle(
	//		&vertices[8], &vertices[11], &vertices[10],
	//		nullptr, nullptr, nullptr,
	//		&normals[2], &normals[2], &normals[2]);
	//	mesh->GetMeshStructure()->CreateTriangle(
	//		&vertices[8], &vertices[10], &vertices[9],
	//		nullptr, nullptr, nullptr,
	//		&normals[2], &normals[2], &normals[2]);

	//	// right
	//	mesh->GetMeshStructure()->CreateTriangle(
	//		&vertices[12], &vertices[15], &vertices[14],
	//		nullptr, nullptr, nullptr,
	//		&normals[3], &normals[3], &normals[3]);
	//	mesh->GetMeshStructure()->CreateTriangle(
	//		&vertices[12], &vertices[14], &vertices[13],
	//		nullptr, nullptr, nullptr,
	//		&normals[3], &normals[3], &normals[3]);

	//	// back
	//	mesh->GetMeshStructure()->CreateTriangle(
	//		&vertices[19], &vertices[18], &vertices[17]);
	//	mesh->GetMeshStructure()->CreateTriangle(
	//		&vertices[19], &vertices[17], &vertices[16]);

	//	// front
	//	mesh->GetMeshStructure()->CreateTriangle(
	//		&vertices[20], &vertices[21], &vertices[22]);
	//	mesh->GetMeshStructure()->CreateTriangle(
	//		&vertices[20], &vertices[22], &vertices[23]);


	//	// corners
	//	mesh->GetMeshStructure()->CreateTriangle(
	//		&vertices[0], &vertices[20], &vertices[8],
	//		nullptr, nullptr, nullptr,
	//		&normals[0], &normals[5], &normals[2]);
	//	mesh->GetMeshStructure()->CreateTriangle(
	//		&vertices[1], &vertices[12], &vertices[21],
	//		nullptr, nullptr, nullptr,
	//		&normals[0], &normals[3], &normals[5]);
	//	mesh->GetMeshStructure()->CreateTriangle(
	//		&vertices[2], &vertices[15], &vertices[19],
	//		nullptr, nullptr, nullptr,
	//		&normals[0], &normals[3], &normals[4]);
	//	mesh->GetMeshStructure()->CreateTriangle(
	//		&vertices[3], &vertices[11], &vertices[16],
	//		nullptr, nullptr, nullptr,
	//		&normals[0], &normals[2], &normals[4]);
	//	mesh->GetMeshStructure()->CreateTriangle(
	//		&vertices[4], &vertices[9], &vertices[23],
	//		nullptr, nullptr, nullptr,
	//		&normals[1], &normals[2], &normals[5]);
	//	mesh->GetMeshStructure()->CreateTriangle(
	//		&vertices[5], &vertices[13], &vertices[22],
	//		nullptr, nullptr, nullptr,
	//		&normals[1], &normals[3], &normals[5]);
	//	mesh->GetMeshStructure()->CreateTriangle(
	//		&vertices[6], &vertices[14], &vertices[18],
	//		nullptr, nullptr, nullptr,
	//		&normals[1], &normals[3], &normals[4]);
	//	mesh->GetMeshStructure()->CreateTriangle(
	//		&vertices[7], &vertices[10], &vertices[17],
	//		nullptr, nullptr, nullptr,
	//		&normals[1], &normals[2], &normals[4]);


	//	// edges
	//	mesh->GetMeshStructure()->CreateTriangle(
	//		&vertices[0], &vertices[1], &vertices[21],
	//		nullptr, nullptr, nullptr,
	//		&normals[0], &normals[0], &normals[5]);
	//	mesh->GetMeshStructure()->CreateTriangle(
	//		&vertices[0], &vertices[21], &vertices[20],
	//		nullptr, nullptr, nullptr,
	//		&normals[0], &normals[5], &normals[5]);
	//	mesh->GetMeshStructure()->CreateTriangle(
	//		&vertices[1], &vertices[2], &vertices[15]);
	//	mesh->GetMeshStructure()->CreateTriangle(
	//		&vertices[1], &vertices[15], &vertices[12]);
	//	mesh->GetMeshStructure()->CreateTriangle(
	//		&vertices[2], &vertices[3], &vertices[16]);
	//	mesh->GetMeshStructure()->CreateTriangle(
	//		&vertices[2], &vertices[16], &vertices[19]);
	//	mesh->GetMeshStructure()->CreateTriangle(
	//		&vertices[3], &vertices[0], &vertices[8]);
	//	mesh->GetMeshStructure()->CreateTriangle(
	//		&vertices[3], &vertices[8], &vertices[11]);

	//	mesh->GetMeshStructure()->CreateTriangle(
	//		&vertices[4], &vertices[5], &vertices[22]);
	//	mesh->GetMeshStructure()->CreateTriangle(
	//		&vertices[4], &vertices[22], &vertices[23]);
	//	mesh->GetMeshStructure()->CreateTriangle(
	//		&vertices[5], &vertices[6], &vertices[14]);
	//	mesh->GetMeshStructure()->CreateTriangle(
	//		&vertices[5], &vertices[14], &vertices[13]);
	//	mesh->GetMeshStructure()->CreateTriangle(
	//		&vertices[6], &vertices[7], &vertices[17]);
	//	mesh->GetMeshStructure()->CreateTriangle(
	//		&vertices[6], &vertices[17], &vertices[18]);
	//	mesh->GetMeshStructure()->CreateTriangle(
	//		&vertices[4], &vertices[7], &vertices[10]);
	//	mesh->GetMeshStructure()->CreateTriangle(
	//		&vertices[4], &vertices[10], &vertices[9]);

	//	mesh->GetMeshStructure()->CreateTriangle(
	//		&vertices[8], &vertices[20], &vertices[23]);
	//	mesh->GetMeshStructure()->CreateTriangle(
	//		&vertices[8], &vertices[23], &vertices[9]);
	//	mesh->GetMeshStructure()->CreateTriangle(
	//		&vertices[12], &vertices[21], &vertices[22]);
	//	mesh->GetMeshStructure()->CreateTriangle(
	//		&vertices[12], &vertices[22], &vertices[13]);
	//	mesh->GetMeshStructure()->CreateTriangle(
	//		&vertices[15], &vertices[19], &vertices[18]);
	//	mesh->GetMeshStructure()->CreateTriangle(
	//		&vertices[15], &vertices[18], &vertices[14]);
	//	mesh->GetMeshStructure()->CreateTriangle(
	//		&vertices[11], &vertices[16], &vertices[17]);
	//	mesh->GetMeshStructure()->CreateTriangle(
	//		&vertices[11], &vertices[17], &vertices[10]);

	//	auto& triangles = mesh->GetMeshStructure()->GetTriangles();
	//	for (uint32_t i = 0; i < triangles.GetCount(); i++)
	//	{
	//		triangles[i].color = Graphics::Color(0x80, 0x80, 0x80);
	//	}
	//}
}